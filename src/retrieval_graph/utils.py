"""Utility functions for the retrieval graph.

Functions:
    get_message_text: Extract text content from various message formats.
    format_docs: Convert documents to an XML-formatted string with content limits.
    format_docs_safe: Like ``format_docs`` but bounded by an estimated token budget.
    format_docs_with_citations: Format documents with title/source/type attributes.
    estimate_tokens: Estimate token count for text (rough approximation).
    truncate_to_token_limit: Truncate text to stay within token limits.
    load_chat_model: Create a chat model from a ``provider/model`` string.
    create_documents_from_urls: Download web pages / PDFs and extract their text.
"""

import asyncio
import io
import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import aiohttp
import pypdf
from bs4 import BeautifulSoup
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

URL_FETCH_TIMEOUT_SECONDS = 60
"""Total time allowed to download one URL before giving up."""


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    This is a rough approximation: 1 token ≈ 4 characters for English text.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    if not text:
        return 0
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int = 150000) -> str:
    """Truncate text to stay within a token limit.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens allowed (default: 150000).

    Returns:
        Truncated text that should be within the token limit.
    """
    if not text:
        return text

    if estimate_tokens(text) <= max_tokens:
        return text

    # Calculate how much to truncate (with some safety margin)
    target_chars = max_tokens * 4 * 0.9  # 90% of estimated limit for safety
    if len(text) > target_chars:
        truncated = text[: int(target_chars)]
        return truncated + "\n\n... [CONTENT TRUNCATED TO PREVENT TOKEN OVERFLOW]"

    return text


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message.

    Handles both plain string content and lists of content blocks (as returned by
    some providers, e.g. Anthropic); non-text blocks such as tool calls are skipped.

    Args:
        msg: The message object to extract text from.

    Returns:
        The extracted text content of the message.

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content=[{"type": "text", "text": "Hello"}, " ", {"type": "text", "text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    texts = [c if isinstance(c, str) else str(c.get("text") or "") for c in content]
    return "".join(texts).strip()


def _format_metadata(metadata: Mapping[str, Any] | None) -> str:
    """Render metadata as XML-ish attributes for a ``<document>`` tag."""
    meta = "".join(f" {k}={v!r}" for k, v in (metadata or {}).items())
    return f" {meta}" if meta else ""


def _format_doc(doc: Document, max_content_length: int = 2000) -> str:
    """Format a single document as XML with content truncation.

    Args:
        doc: The document to format.
        max_content_length: Maximum length of content to include (default: 2000 chars).

    Returns:
        The formatted document as an XML string.
    """
    content = doc.page_content
    if len(content) > max_content_length:
        content = content[:max_content_length] + "... [TRUNCATED]"
    return f"<document{_format_metadata(doc.metadata)}>\n{content}\n</document>"


def format_docs_safe(docs: list[Document] | None, max_total_tokens: int = 50000) -> str:
    """Format documents with intelligent token-based truncation.

    This function formats documents while staying within token limits by:
    1. Limiting the number of documents
    2. Truncating individual document content
    3. Monitoring total token usage

    Args:
        docs: Documents to format.
        max_total_tokens: Maximum total tokens for all documents (default: 50000).

    Returns:
        Formatted documents within token limits.
    """
    if not docs:
        return "<documents></documents>"

    formatted_docs = []
    total_tokens = 0
    max_doc_tokens = 8000  # Max tokens per document

    for i, doc in enumerate(docs):
        # Truncate individual document content to prevent any single doc from being too large
        content = doc.page_content
        doc_tokens = estimate_tokens(content)

        if doc_tokens > max_doc_tokens:
            target_chars = max_doc_tokens * 4 * 0.9
            content = content[: int(target_chars)] + "... [DOCUMENT TRUNCATED]"
            doc_tokens = estimate_tokens(content)

        # Check if adding this document would exceed our total limit
        if total_tokens + doc_tokens > max_total_tokens:
            if i == 0:
                # If even the first document would exceed limits, truncate it more aggressively
                remaining_tokens = max_total_tokens - 1000  # Leave room for XML markup
                target_chars = remaining_tokens * 4 * 0.9
                content = (
                    doc.page_content[: int(target_chars)]
                    + "... [HEAVILY TRUNCATED DUE TO SIZE]"
                )
            else:
                # Stop adding documents, we've reached our limit
                break

        formatted_doc = (
            f"<document{_format_metadata(doc.metadata)}>\n{content}\n</document>"
        )
        formatted_docs.append(formatted_doc)
        total_tokens += estimate_tokens(formatted_doc)

    formatted = "\n".join(formatted_docs)

    truncation_note = ""
    if len(formatted_docs) < len(docs):
        truncation_note = f"\n<!-- Note: Showing {len(formatted_docs)} of {len(docs)} documents (others truncated to prevent token overflow) -->"

    result = f"""<documents>
{formatted}{truncation_note}
</documents>"""

    # Final safety check
    if estimate_tokens(result) > max_total_tokens:
        result = truncate_to_token_limit(result, max_total_tokens)

    return result


def format_docs(
    docs: list[Document] | None, max_content_length: int = 2000, max_total_docs: int = 3
) -> str:
    """Format a list of documents as XML with content and document limits.

    This function takes a list of Document objects and formats them into a single XML string,
    with content truncation and document count limits to prevent token overflow.

    Args:
        docs: A list of Document objects to format, or None.
        max_content_length: Maximum length of content per document (default: 2000 chars).
        max_total_docs: Maximum number of documents to include (default: 3).

    Returns:
        A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"

    limited_docs = docs[:max_total_docs]
    formatted = "\n".join(_format_doc(doc, max_content_length) for doc in limited_docs)

    truncation_note = ""
    if len(docs) > max_total_docs:
        truncation_note = f"\n<!-- Note: Showing {max_total_docs} of {len(docs)} documents to prevent token overflow -->"

    return f"""<documents>
{formatted}{truncation_note}
</documents>"""


def format_docs_with_citations(docs: list[Document] | None) -> str:
    """Format documents with enhanced citation information.

    Args:
        docs: Documents to format.

    Returns:
        Formatted documents with citation metadata.
    """
    if not docs:
        return "<documents></documents>"

    formatted = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        source = metadata.get("source", f"Document_{i}")
        title = metadata.get("title", f"Document {i}")
        doc_type = metadata.get("type", "unknown")

        citation_info = f'title="{title}" source="{source}" type="{doc_type}"'
        formatted.append(f"<document {citation_info}>\n{doc.page_content}\n</document>")

    body = "\n".join(formatted)
    return f"<documents>\n{body}\n</documents>"


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name: String in the format 'provider/model'. Without a
            provider prefix, LangChain infers the provider from the model name.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider, model = "", fully_specified_name
    return init_chat_model(model, model_provider=provider or None)


######################
# URL ingestion
######################


def _pdf_bytes_to_text(data: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(data))
    return "".join(page.extract_text() + "\n" for page in reader.pages).strip()


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    lines = (line.strip() for line in soup.get_text().splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(chunk for chunk in chunks if chunk)


def _client_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=URL_FETCH_TIMEOUT_SECONDS)


async def extract_text_from_pdf_url(pdf_url: str) -> str:
    """Extract text content from a PDF URL.

    Args:
        pdf_url: URL pointing to a PDF file.

    Returns:
        Extracted text content from the PDF, or a message starting with "Failed"
        or "Error" describing what went wrong.
    """
    try:
        async with (
            aiohttp.ClientSession(timeout=_client_timeout()) as session,
            session.get(pdf_url) as response,
        ):
            if response.status != 200:
                return f"Failed to download PDF: HTTP {response.status}"
            pdf_content = await response.read()
        # Parsing is CPU-bound; keep it off the event loop.
        return await asyncio.to_thread(_pdf_bytes_to_text, pdf_content)
    except TimeoutError:
        return f"Error processing PDF: timed out after {URL_FETCH_TIMEOUT_SECONDS}s"
    except Exception as e:
        return f"Error processing PDF: {e!s}"


async def extract_text_from_web_url(web_url: str) -> str:
    """Extract text content from a web URL.

    Args:
        web_url: URL pointing to a web page.

    Returns:
        Extracted text content from the web page, or a message starting with
        "Failed" or "Error" describing what went wrong.
    """
    try:
        async with (
            aiohttp.ClientSession(timeout=_client_timeout()) as session,
            session.get(web_url) as response,
        ):
            if response.status != 200:
                return f"Failed to access URL: HTTP {response.status}"
            html_content = await response.text()
        return await asyncio.to_thread(_html_to_text, html_content)
    except TimeoutError:
        return (
            f"Error processing web page: timed out after {URL_FETCH_TIMEOUT_SECONDS}s"
        )
    except Exception as e:
        return f"Error processing web page: {e!s}"


def _error_document(url: str, content: str, title: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": url,
            "type": "error",
            "title": title,
            "extraction_success": False,
            "content_length": 0,
        },
    )


async def create_documents_from_urls(urls: list[str]) -> list[Document]:
    """Create Document objects from a list of URLs with content validation.

    URLs are fetched concurrently. URLs whose path ends in ``.pdf`` are parsed as
    PDFs, everything else as HTML. One document is returned per URL, in order; a
    failed URL yields a document whose ``metadata["extraction_success"]`` is False.

    Args:
        urls: List of URLs to process.

    Returns:
        List of Document objects with extracted content or error information.
    """

    async def process_url(url: str) -> Document:
        """Process a single URL and always return a Document (with error info if needed)."""
        try:
            parsed_url = urlparse(url)
            if parsed_url.path.lower().endswith(".pdf"):
                content = await extract_text_from_pdf_url(url)
                doc_type = "pdf"
            else:
                content = await extract_text_from_web_url(url)
                doc_type = "webpage"

            extraction_success = not (
                content.startswith(
                    ("Failed", "Error", "Document processing dependencies")
                )
                or content == url
                or len(content.strip()) < 100
            )
            return Document(
                page_content=content,
                metadata={
                    "source": url,
                    "type": doc_type,
                    "title": parsed_url.path.split("/")[-1] or parsed_url.netloc,
                    "extraction_success": extraction_success,
                    "content_length": len(content.strip()),
                },
            )
        except Exception as e:
            logger.error("Error processing %s: %s", url, e)
            return _error_document(
                url, f"Error processing URL: {e!s}", "Processing Error"
            )

    results = await asyncio.gather(
        *(process_url(url) for url in urls), return_exceptions=True
    )

    documents = []
    for url, result in zip(urls, results, strict=True):
        if isinstance(result, Document):
            documents.append(result)
        else:  # Exceptions that escaped process_url (e.g. cancellation).
            documents.append(
                _error_document(
                    url,
                    f"Exception during processing: {result!s}",
                    "Processing Exception",
                )
            )
    return documents
