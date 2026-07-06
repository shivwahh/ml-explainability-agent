"""
project_context_loader.py

Ingestion loader for the project context artifact.

Reads unstructured business knowledge from ``.md`` and ``.txt`` files
and exposes an extensible reader interface so additional formats such as
``.pdf`` and ``.docx`` can be plugged in without changing the loader.
"""

from pathlib import Path


def _read_text_file(path: Path) -> str:
    """
    Read a plain text or markdown file as UTF-8 text.
    """
    return path.read_text(encoding="utf-8")


class ProjectContextLoader:
    """
    Load unstructured project context and split it into sections.

    Readers are keyed by file extension. Built-in readers handle ``.md``
    and ``.txt``. Additional readers (for example ``.pdf`` or ``.docx``)
    can be registered via :meth:`register_reader`.
    """

    def __init__(self, context_path: str):
        """
        Args:
            context_path: Path to the project context file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self.context_path = Path(context_path)

        if not self.context_path.exists():
            raise FileNotFoundError(
                f"Project context not found: {self.context_path}"
            )

        self._readers = {
            ".md": _read_text_file,
            ".txt": _read_text_file,
        }

        self.text = None

    def register_reader(self, extension: str, reader) -> None:
        """
        Register a reader for a file extension.

        Args:
            extension: File extension including the leading dot
                (for example ``".pdf"``).
            reader: A callable accepting a :class:`pathlib.Path` and
                returning the file contents as a string.
        """
        self._readers[extension.lower()] = reader

    def supported_extensions(self) -> list:
        """
        Return the currently supported extensions.
        """
        return sorted(self._readers)

    def load(self) -> str:
        """
        Load and return the raw project context text.

        Returns:
            The raw file contents as a string.

        Raises:
            ValueError: If no reader is registered for the extension.
        """
        suffix = self.context_path.suffix.lower()

        reader = self._readers.get(suffix)

        if reader is None:
            raise ValueError(
                f"No reader registered for '{suffix}'. "
                f"Register one with register_reader() or use a "
                f"supported extension: {self.supported_extensions()}"
            )

        self.text = reader(self.context_path)

        return self.text

    def to_sections(self) -> dict:
        """
        Split the context into sections keyed by markdown heading.

        Content that appears before the first heading is stored under the
        ``"_preamble"`` key. The context is loaded automatically if it has
        not been loaded yet.

        Returns:
            A mapping of heading text to the section body.
        """
        if self.text is None:
            self.load()

        sections = {}
        current_heading = "_preamble"
        current_lines = []

        for line in self.text.splitlines():
            stripped = line.strip()

            if stripped.startswith("#"):
                sections[current_heading] = "\n".join(
                    current_lines
                ).strip()

                current_heading = stripped.lstrip("#").strip()
                current_lines = []
            else:
                current_lines.append(line)

        sections[current_heading] = "\n".join(current_lines).strip()

        return {
            heading: body
            for heading, body in sections.items()
            if body or heading != "_preamble"
        }
