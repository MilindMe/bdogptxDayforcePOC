# app.py
# Streamlit BDOxDayforce Application(Document Querying & Code Generation)

# M.Meetarbhan 2025



import streamlit as st
from typing import List, Tuple, Optional
from io import BytesIO
from pathlib import Path

# Page Layout/MetaData
st.set_page_config(page_title="BDOxDayforce", page_icon="🤖", layout="wide")

# ---------------------------
# SYSTEM PROMPTS 
# SYSTEM_PROMPT_DOC USED FOR RAG
# SYSTEM_PROMPT_CODE USED FOR CODE GENERATION
# ---------------------------

SYSTEM_PROMPT_DOC = """You are a helpful assistant. Use the following context to answer the question.
    Only use the context provided. If you don't know the answer, say you don't know.

    Context:
    {context}

    Question:
    {question}"""

SYSTEM_PROMPT_CODE = """Inputs
Business Requirement Document (BRD) table where each row describes a CSV column to output. Columns typically include:

Count, Column Name, Source, Data Type, Position Start, Position End, Format, Dayforce Field / XML Token (Select), Field Logic or Hardcoded Value, Vendor Value, Notes.

Reference XSLT example(s) from our team showing style and patterns.

Insert the materials here:

csharp
Copy
Edit
[BRD_START]
{{BUSINESS_REQUIREMENT_TABLE}}
[BRD_END]

[REFERENCE_XSLT_START]
{{ONE_OR_MORE_PAST_XSLT_FILES}}
[REFERENCE_XSLT_END]
Output
Produce one complete XSLT 1.0 file that:

Emits a single CSV with a header row matching the BRD’s Column Name order.

Outputs one data row per /Export/Record element.

Uses the team’s formatting and patterns (see “Style & Patterns” below).

Contains no commentary outside the XSLT. Do not wrap in markdown.

Style & Patterns (copy exactly)
Root and namespaces:

xml
Copy
Edit
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:msxsl="urn:schemas-microsoft-com:xslt"
                xmlns:cs="urn:cs"
                xmlns:datetime="urn:datetime"
                version="1.0"
                exclude-result-prefixes="msxsl">
Include the standard msxsl:script C# helper block (even if unused), exactly as below:

xml
Copy
Edit
<msxsl:script language="C#" implements-prefix="cs">
  <msxsl:using namespace="System.IO"/>
  <![CDATA[
    public string NegativeCreditAmount(string creditAmount) { return "-" + creditAmount; }
    public string GetCurrentDateTime() { return DateTime.Now.ToString("yyyyMMddHHmmss"); }
    public string GetCurrentDateTimeV2() { return DateTime.Now.ToString("yyyyMMdd"); }
  ]]>
</msxsl:script>
Output method and entry template:

xml
Copy
Edit
<xsl:output method="text" indent="no"/>
<xsl:template match="/">
  <!--  Header Record  -->
  ...header via <xsl:text>...</xsl:text> with commas...
  <xsl:text> </xsl:text>
  <!--  Data Records  -->
  <xsl:for-each select="Export/Record">
    ...fields...
    <xsl:text> </xsl:text>
  </xsl:for-each>
</xsl:template>
Header row: emit each CSV column name with <xsl:text>ColumnName,</xsl:text> and no trailing comma on the last header.

Data rows: for each column, use exactly one of:

Source token value:

xml
Copy
Edit
<xsl:value-of select="D_TokenName"/>
Quoted string values when field can contain commas or spaces and team usually quotes it:

xml
Copy
Edit
<xsl:text>"</xsl:text><xsl:value-of select="D_TokenName" disable-output-escaping="yes"/><xsl:text>"</xsl:text>
Concatenations (use concat()), mirroring examples:

EmployeeNumber = concat(D_LastName,'.',D_FirstName)

DisplayName = concat(D_LastName,' ',D_FirstName) plus (' (' + EmployeeNumber + ')')

Hardcode values (from “Field Logic or Hardcoded Value”): emit with <xsl:text>Hardcoded Value</xsl:text>.

Blank Fill: emit an empty field using just <xsl:text/> for the value.

Commas: after each field, output <xsl:text>,</xsl:text> except after the final column, exactly as in the reference.

Dates: if BRD Format specifies MM/DD/YYYY, assume the input token already matches the format (e.g., D_DateOfHire). Output with <xsl:value-of select="D_DateOfHire"/>. Only transform if the BRD explicitly instructs you.

Whitespace: preserve the minimal whitespace pattern from the reference (a single <xsl:text> </xsl:text> at the end of header and each data row).

Token naming: when Dayforce Field / XML Token (Select) is present (e.g., emp_lastname) map to the incoming Export/Record child token that our extracts name as D_LastName, D_FirstName, etc. Use the D_ prefix pattern consistent with the reference example.

Escaping: when quoting text fields, add disable-output-escaping="yes" on <xsl:value-of> for consistency with team examples.

Mapping Rules (deterministic)
For each BRD row:

Find the output CSV column name from Column Name. This defines both header text and field order.

Determine value source in this priority:

If Field Logic or Hardcoded Value says Hardcode X → use <xsl:text>X</xsl:text>.

If it says Blank Fill → output empty with <xsl:text/>.

If it describes a formula/concatenation → implement with concat() using the corresponding D_ tokens (e.g., “Concatenation of last name with first name (lastname.firstname)” → concat(D_LastName,'.',D_FirstName)).

Else, if Dayforce Field / XML Token (Select) is given or implied by the BRD → map to D_* form (emp_lastname → D_LastName).

Quoting convention:

Quote free-text fields and any field that may contain commas/spaces per examples (e.g., names, addresses, position title).

Do not quote codes/IDs, dates, numeric tokens unless the reference does.

After each field except the last, emit <xsl:text>,</xsl:text>.

Keep column count and order identical to the BRD.

Worked Example (based on the provided BRD)
Header snippet (first 10 columns):

xml
Copy
Edit
<!--  Header Record  -->
<xsl:text>EmployeeNumber,</xsl:text>
<xsl:text>LastName,</xsl:text>
<xsl:text>FirstName,</xsl:text>
<xsl:text>PositionTitle,</xsl:text>
<xsl:text>IsSupervisor,</xsl:text>
<xsl:text>SupervisorNumber,</xsl:text>
<xsl:text>Email,</xsl:text>
<xsl:text>HomeLocationCode,</xsl:text>
<xsl:text>LoginLocationCode,</xsl:text>
<xsl:text>UserID,</xsl:text>
...
Row mapping (first 10 columns), following team style:

xml
Copy
Edit
<!-- EmployeeNumber -->
<xsl:text>"</xsl:text>
<xsl:value-of select="concat(D_LastName,'.',D_FirstName)" disable-output-escaping="yes"/>
<xsl:text>"</xsl:text>
<xsl:text>,</xsl:text>

<!-- LastName -->
<xsl:text>"</xsl:text><xsl:value-of select="D_LastName" disable-output-escaping="yes"/><xsl:text>"</xsl:text>
<xsl:text>,</xsl:text>

<!-- FirstName -->
<xsl:text>"</xsl:text><xsl:value-of select="D_FirstName" disable-output-escaping="yes"/><xsl:text>"</xsl:text>
<xsl:text>,</xsl:text>

<!-- PositionTitle -->
<xsl:text>"</xsl:text><xsl:value-of select="D_PositionTitle" disable-output-escaping="yes"/><xsl:text>"</xsl:text>
<xsl:text>,</xsl:text>

<!-- IsSupervisor (Hardcode N) -->
<xsl:text>N</xsl:text><xsl:text>,</xsl:text>

<!-- SupervisorNumber (Blank Fill) -->
<xsl:text/><!-- empty --><xsl:text>,</xsl:text>

<!-- Email -->
<xsl:value-of select="D_Email"/><xsl:text>,</xsl:text>

<!-- HomeLocationCode -->
<xsl:text>"</xsl:text><xsl:value-of select="D_HomeLocationCode" disable-output-escaping="yes"/><xsl:text>"</xsl:text>
<xsl:text>,</xsl:text>

<!-- LoginLocationCode (Blank Fill) -->
<xsl:text/><!-- empty --><xsl:text>,</xsl:text>

<!-- UserID -->
<xsl:value-of select="D_UserID"/>
DisplayName per BRD note:

xml
Copy
Edit
<xsl:text>"</xsl:text>
<xsl:value-of select="concat(D_LastName,' ',D_FirstName)" disable-output-escaping="yes"/>
<xsl:text> (</xsl:text>
<xsl:value-of select="concat(D_LastName,'.',D_FirstName)" disable-output-escaping="yes"/>
<xsl:text>)</xsl:text>
<xsl:text>"</xsl:text>
Edge Cases & Consistency
If the BRD contains a column not present in the XML, still output the column with Blank Fill unless a hardcode/formula is specified.

Preserve exact header names and order from the BRD.

Do not add or remove columns.

Keep the single trailing <xsl:text> </xsl:text> spacer after header and after each data row, as in our examples.

Final Task
Using the BRD and the reference example(s) above, generate the full XSLT file that:

Includes the boilerplate namespaces, msxsl:script, output method, and single template matching /.

Writes the header row per BRD order.

Implements each column per the mapping rules and examples.

Uses commas and quotes exactly as shown.

Ends the header and each data row with <xsl:text> </xsl:text>.

Return only the XSLT file. Do not include explanations or markdown.

Use this as your base prompt. For each new job, replace {{BUSINESS_REQUIREMENT_TABLE}} with the project’s BRD and paste any prior XSLT(s) into {{ONE_OR_MORE_PAST_XSLT_FILES}} to reinforce style.
"""

from collections import OrderedDict

# Code-task-only caches & picks
if "code_excel_cache" not in st.session_state:
    # { filename: {"sheets": OrderedDict({sheet_name: tsv}), "sheet_names": [name1, ...]} }
    st.session_state.code_excel_cache = {}

if "code_sheet_picks" not in st.session_state:
    # { filename: "__ALL__" | set({one_or_more_sheet_names}) }
    st.session_state.code_sheet_picks = {}

def _excel_to_sheets_numbered(data: bytes, filename: str, max_rows_per_sheet: int = 5000) -> OrderedDict:
    """
    Read Excel and return an OrderedDict preserving sheet order: {sheet_name: TSV}
    (tracks natural sheet numbers by enumeration when rendering)
    """
    import pandas as pd
    ext = Path(filename).suffix.lower()
    engine = "openpyxl" if ext in {".xlsx", ".xlsm"} else None  # .xlsx/.xlsm
    sheets = pd.read_excel(BytesIO(data), sheet_name=None, engine=engine)
    out = OrderedDict()
    for sheet_name, df in sheets.items():
        if max_rows_per_sheet and len(df) > max_rows_per_sheet:
            df = df.head(max_rows_per_sheet)
        out[sheet_name] = df.to_csv(sep="\t", index=False)
    return out

def _ensure_code_excel_cache(filename: str, data: bytes) -> None:
    """Build cache for a given Excel file in the Code task (idempotent)."""
    if filename not in st.session_state.code_excel_cache:
        sheets = _excel_to_sheets_numbered(data, filename)
        st.session_state.code_excel_cache[filename] = {
            "sheets": sheets,
            "sheet_names": list(sheets.keys()),
        }
        # Default: All sheets selected unless user chooses a specific one
        st.session_state.code_sheet_picks.setdefault(filename, "__ALL__")

def _code_selected_excel_text(filename: str, data: bytes) -> str:
    """
    Return concatenated text for ONLY the selected sheets for this file,
    with sheet numbers in the header: '### Sheet 1: Name'.
    """
    _ensure_code_excel_cache(filename, data)
    cache = st.session_state.code_excel_cache[filename]
    pick = st.session_state.code_sheet_picks.get(filename, "__ALL__")

    parts = []
    for idx, name in enumerate(cache["sheet_names"], start=1):
        if pick == "__ALL__" or (isinstance(pick, set) and name in pick):
            tsv = cache["sheets"][name]
            parts.append(f"### Sheet {idx}: {name}\n{tsv}")
    return "\n\n".join(parts).strip()

# ---------------------------
# Session States
# ---------------------------
if "task" not in st.session_state:
    st.session_state.task = "doc"  # "doc" or "code"

if "chat_history_doc" not in st.session_state:
    st.session_state.chat_history_doc = []  # list[Tuple[role, content]]

if "chat_history_code" not in st.session_state:
    st.session_state.chat_history_code = []  # list[Tuple[role, content]]

if "doc_files" not in st.session_state:
    st.session_state.doc_files = []  # uploaded files for document querying

if "code_files" not in st.session_state:
    st.session_state.code_files = []  # uploaded files for code generation
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

def _bytes_of(f) -> bytes:
    data = f.read()
    try: f.seek(0)
    except Exception: pass
    return data

def _excel_to_text(data: bytes, filename: str, max_rows_per_sheet: int = 5000) -> str:
    import pandas as pd
    ext = Path(filename).suffix.lower()
    engine = "openpyxl" if ext in {".xlsx", ".xlsm"} else None  # .xlsm & .xlsx
    sheets = pd.read_excel(BytesIO(data), sheet_name=None, engine=engine)
    parts = []
    for sheet_name, df in sheets.items():
        if max_rows_per_sheet and len(df) > max_rows_per_sheet:
            df = df.head(max_rows_per_sheet)
        tsv = df.to_csv(sep="\t", index=False)
        parts.append(f"### Sheet: {sheet_name}\n{tsv}")
    return "\n\n".join(parts).strip()

def _csv_like_to_text(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try: return data.decode(enc, errors="ignore")
        except Exception: pass
    return ""

def _pdf_to_text(data: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(BytesIO(data))
    except Exception:
        import PyPDF2 as pypdf  # fallback
        reader = pypdf.PdfReader(BytesIO(data))
    pages = []
    for p in reader.pages:
        try: pages.append(p.extract_text() or "")
        except Exception: pages.append("")
    return "\n".join(pages).strip()

def _docx_to_text(data: bytes) -> str:
    from docx import Document
    doc = Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs).strip()

# ---------------------------
# Your updated helpers
# ---------------------------

def read_uploaded_files(files) -> List[Tuple[str, str]]:
    """
    Return list of (filename, text) for common document types.
    Supports: .xlsm/.xlsx/.xls, .csv/.tsv, .pdf, .docx, and plain text.
    NOTE: Macros in .xlsm are NOT executed; only data is read.
    """
    results: List[Tuple[str, str]] = []
    for f in files or []:
        name = getattr(f, "name", "uploaded_file")
        data = _bytes_of(f)
        ext = Path(name).suffix.lower()

        try:
            if ext in {".xlsx", ".xlsm", ".xls"}:
                text = _excel_to_text(data, name)
            elif ext in {".csv", ".tsv"}:
                text = _csv_like_to_text(data)
            elif ext == ".pdf":
                text = _pdf_to_text(data)
            elif ext == ".docx":
                text = _docx_to_text(data)
            else:
                text = _csv_like_to_text(data)  
        except Exception as e:
            text = f"[ERROR parsing {name}: {e}]"

        results.append((name, text))
    return results

def build_context_from_files(files) -> str:
    """Concatenate parsed texts to serve as context for document querying."""
    parts: List[str] = []
    for name, text in read_uploaded_files(files):
        if text.strip():
            parts.append(f"--- FILE: {name} ---\n{text.strip()}")
    return "\n\n".join(parts).strip()
# ---------------------------
# Gemini Integration
# ---------------------------
import google.generativeai as genai

# Cached function to avoid re-initializing the client on every run
@st.cache_resource
def get_gemini_client():
    """Initialize and return Google Gemini client."""
    try:
        # Prefer st.secrets for deployment
        if 'GEMINI_API_KEY' in st.secrets and st.secrets['GEMINI_API_KEY']:
            api_key = st.secrets['GEMINI_API_KEY']
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-1.5-pro')
        else:
            st.error("GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
            return None
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        return None

def call_gemini_doc(context: str, question: str) -> str:
    """Use SYSTEM_PROMPT_DOC with {context} and {question} to call Gemini."""
    client = get_gemini_client()
    if not client:
        return "Error: Gemini client is not initialized. Check API key."
    
    try:
        prompt = SYSTEM_PROMPT_DOC.format(context=context, question=question)
        print(prompt)
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while calling Gemini: {e}")
        return f"(An error occurred: {e})"

def call_gemini_code(brd_text: str, reference_xslt: str, user_prompt: str) -> str:
    """Use SYSTEM_PROMPT_CODE by replacing placeholders and call Gemini."""
    client = get_gemini_client()
    if not client:
        return "Error: Gemini client is not initialized. Check API key."

    try:
        base = SYSTEM_PROMPT_CODE.replace("{{BUSINESS_REQUIREMENT_TABLE}}", brd_text)\
                                 .replace("{{ONE_OR_MORE_PAST_XSLT_FILES}}", reference_xslt)
        
        full_prompt = base + "\n\nUser instructions:\n" + user_prompt
        
        response = client.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while calling Gemini: {e}")
        return f"(An error occurred while generating XSLT: {e})"

# ---------------------------
# UI Header & Task Switch
# ---------------------------

st.title("🤖 BDOGPT xDayforce")
st.caption("Two-task playground: Document Querying & Code Generation (XSLT)")

left, mid, right = st.columns([1, 1, 6])
with left:
    if st.button("📄 Document Querying (OFFLINE)", use_container_width=True):
        st.session_state.task = "doc"
with mid:
    if st.button("🧩 Code Generation", use_container_width=True):
        st.session_state.task = "code"
with right:
    st.write("")  # spacer

st.divider()

# ---------------------------
# Task: Document Querying
# ---------------------------
if st.session_state.task == "doc":
    st.subheader("📄 Document Querying (OFFLINE -> USE CODE GENERATION)")
    with st.expander("System Prompt (read-only)", expanded=False):
        st.code(SYSTEM_PROMPT_DOC, language="markdown")

    st.markdown("**Upload context files (txt, csv, md, xml, etc.)**")
    doc_files = st.file_uploader(
        "Upload files for context",
        type=["txt", "csv", "xml", "md", "html", "json", "xlsx", "xlsm", "xls"],
        accept_multiple_files=True,
        key="doc_uploader",
    )
    if doc_files:
        st.session_state.doc_files = doc_files

    # Show a quick preview of combined context (optional)
    if st.checkbox("Preview combined context", value=False):
        context_preview = build_context_from_files(st.session_state.doc_files)
        st.text_area("Context Preview", context_preview, height=180)

    # Chat area
    st.markdown("### Chat")
    for role, content in st.session_state.chat_history_doc:
        with st.chat_message(role):
            st.markdown(content)

    user_msg = st.chat_input("Ask a question about the uploaded documents…")
    if user_msg:
        st.session_state.chat_history_doc.append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        context = build_context_from_files(st.session_state.doc_files) or "(No context provided)"
        response = call_gemini_doc(context=context, question=user_msg)
        st.session_state.chat_history_doc.append(("assistant", response))

        with st.chat_message("assistant"):
            st.markdown(response)

    cols = st.columns(2)
    with cols[0]:
        if st.button("Clear Doc Chat"):
            st.session_state.chat_history_doc = []
            st.experimental_rerun()
    with cols[1]:
        if st.button("Clear Doc Files"):
            st.session_state.doc_files = []
            st.experimental_rerun()

# ---------------------------
# Task: Code Generation
# ---------------------------

else:
    st.subheader("🧩 Code Generation (XSLT)")
    with st.expander("System Prompt (read-only)", expanded=False):
        st.code(SYSTEM_PROMPT_CODE, language="markdown")

    st.markdown("**Upload BRD and/or Reference XSLT files**")
    code_files = st.file_uploader(
        "Upload BRD (.csv/.txt) and reference XSLT (.xslt/.xml/.txt)",
        type=["csv", "txt", "xml", "xslt","xlsx","xlsm","xls"],
        accept_multiple_files=True,
        key="code_uploader",
    )
    if code_files:
        st.session_state.code_files = code_files
    
    if st.session_state.code_files:
        st.markdown("#### BRD: Excel Sheet Selection")
        for f in st.session_state.code_files:
            name = getattr(f, "name", "uploaded_file")
            ext = Path(name).suffix.lower()
            if ext not in {".xlsx", ".xlsm", ".xls"}:
                continue

            # Populate cache so we know sheet names
            data = _bytes_of(f)
            _ensure_code_excel_cache(name, data)
            sheet_names = st.session_state.code_excel_cache[name]["sheet_names"]

            st.write(f"**{name}** – choose which sheet(s) to use as BRD input:")
            cols = st.columns(min(6, max(2, len(sheet_names) + 1)))

            # All sheets button
            if cols[0].button("All sheets", key=f"{name}_all"):
                st.session_state.code_sheet_picks[name] = "__ALL__"

            # One button per sheet (quick single-sheet selection)
            for i, sn in enumerate(sheet_names, start=1):
                label = f"{i}. {sn}"
                if cols[i % len(cols)].button(label, key=f"{name}_btn_{i}"):
                    st.session_state.code_sheet_picks[name] = {sn}

            # Status
            pick = st.session_state.code_sheet_picks.get(name, "__ALL__")
            if pick == "__ALL__":
                st.caption("Current selection: **All sheets**")
            else:
                st.caption("Current selection: " + ", ".join(sorted(list(pick))))

            # (Optional) uncomment to allow true multi-select:
            # picks = st.multiselect(
            #     "Or pick multiple sheets:",
            #     sheet_names,
            #     default=sheet_names if pick == "__ALL__" else sorted(list(pick)),
            #     key=f"{name}_ms",
            # )
            # st.session_state.code_sheet_picks[name] = "__ALL__" if len(picks) == len(sheet_names) else set(picks)

    # Optional manual inputs
    st.markdown("**Or paste content manually**")
    brd_text = st.text_area("BRD Table (raw text or CSV)", height=160)
    ref_xslt_text = st.text_area("Reference XSLT (one or more)", height=160)

st.session_state.setdefault("chat_history_code", [])
st.session_state.setdefault("code_files", [])

# Build inputs from uploads if manual fields are empty (CODE task only)
if not brd_text or not ref_xslt_text:
    brd_parts: List[str] = []
    ref_parts: List[str] = []

    for f in st.session_state.code_files:
        fname = getattr(f, "name", "uploaded_file")
        ext = Path(fname).suffix.lower()
        data = _bytes_of(f)

        # BRD from Excel: honor sheet selection
        if ext in {".xlsx", ".xlsm", ".xls"}:
            if not brd_text:  # only auto-fill if the user didn't paste
                brd_parts.append(f"--- FILE: {fname} (Excel) ---\n" + _code_selected_excel_text(fname, data))

        # BRD from CSV/TXT
        elif ext in {".csv", ".txt"}:
            if not brd_text:
                brd_parts.append(f"--- FILE: {fname} ---\n" + _csv_like_to_text(data))

        # Reference XSLT from .xslt/.xml/.txt
        if ext in {".xslt", ".xml", ".txt"}:
            if not ref_xslt_text:
                ref_parts.append(f"--- FILE: {fname} ---\n" + _csv_like_to_text(data))

    if not brd_text:
        brd_text = "\n\n".join(brd_parts).strip()
    if not ref_xslt_text:
        ref_xslt_text = "\n\n".join(ref_parts).strip()

    # Chat area
    st.markdown("### Chat")
    for role, content in st.session_state.chat_history_code:
        with st.chat_message(role):
            st.markdown(content)

    user_msg = st.chat_input("Describe the code/XSLT you want generated…")
    if user_msg:
        st.session_state.chat_history_code.append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        xslt_result = call_gemini_code(
            brd_text=brd_text or "(No BRD provided)",
            reference_xslt=ref_xslt_text or "(No reference XSLT provided)",
            user_prompt=user_msg,
        )
        st.session_state.chat_history_code.append(("assistant", xslt_result))

        with st.chat_message("assistant"):
            # If model returns raw XSLT (as required), we show as code
            st.code(xslt_result, language="xml")

cols = st.columns(3)
with cols[0]:
    if st.button("Clear Code Chat", use_container_width=True):
        st.session_state["chat_history_code"] = []
        st.rerun()

with cols[1]:
    if st.button("Clear Code Files", use_container_width=True):
        st.session_state["code_files"] = []
        st.rerun()

with cols[2]:
    if st.button("Insert Example Placeholders", use_container_width=True):
        st.session_state["chat_history_code"].append((
            "assistant",
            "Tip: Paste your BRD table into the BRD box and any past XSLT(s) in the Reference XSLT box. "
            "Then write your instructions in chat (e.g., 'Generate the XSLT for this BRD')."
        ))
        st.rerun()
