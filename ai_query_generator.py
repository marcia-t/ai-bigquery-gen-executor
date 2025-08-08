"""
AI Query Generator Module

Handles AI-powered SQL query generation using Google Gemini.
Provides natural language to SQL conversion with context awareness.
"""

import os
from typing import Dict, Optional, Tuple

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class AIQueryGenerator:
    """AI-powered SQL query generator using Google Gemini"""

    def __init__(self, project_id: str = "elastic-observability", metadata_cache=None):
        self.project_id = project_id
        self.metadata_cache = metadata_cache
        self.model = None

        if GEMINI_AVAILABLE:
            try:
                # Get API key from environment
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel("gemini-1.5-flash")
                    print("🤖 AI query generator enabled (Google Gemini)")
                else:
                    print("⚠️  No GEMINI_API_KEY or GOOGLE_API_KEY found")
            except Exception as e:
                print(f"⚠️  Failed to initialize Gemini: {e}")

    def extract_table_reference(self, question: str) -> Tuple[str, str]:
        """Extract dataset and table reference from a question"""
        # Simple pattern matching for dataset.table references
        import re

        pattern = r"\b(\w+)\.(\w+)\b"
        match = re.search(pattern, question)
        if match:
            return match.group(1), match.group(2)
        return "", ""

    def generate_sql_with_ai(
        self,
        question: str,
        context: str = "",
        schemas: Dict[str, Dict] = None,
        max_retries: int = 2,
    ) -> Optional[str]:
        """Generate SQL query using AI with context and schema information"""
        if not self.model:
            return None

        # Build schema context if available
        schema_context = ""
        if schemas:
            schema_context = self._build_schema_context(schemas)

        # Enhanced prompt with schema information
        prompt = f"""
You are a BigQuery SQL expert. Generate a valid BigQuery SQL query for the following question.

IMPORTANT RULES:
1. Always use fully qualified table names: `{self.project_id}.dataset.table`
2. Use proper BigQuery syntax and functions
3. If the question is ambiguous or mentions non-existent tables, return an error message starting with "❌"
4. For cross-dataset table searches, use UNION ALL to combine results from multiple datasets
5. Always add LIMIT clauses for safety (default: 1000 for data queries, no limit for metadata queries)
6. Use backticks around table/column names that might be reserved words
7. ONLY use column names that exist in the provided table schemas

{schema_context}

Available Context:
{context}

Question: {question}

Examples of good responses:
- For "show me tables": SELECT table_name FROM `{self.project_id}.dataset.INFORMATION_SCHEMA.TABLES` LIMIT 1000
- For "count rows": SELECT COUNT(*) FROM `{self.project_id}.dataset.table`
- For ambiguous questions: "❌ Please specify the dataset and table name (e.g., SELECT * FROM `{self.project_id}.dataset.table`)"

Generate only the SQL query or error message. Do not include explanations unless it's an error.
"""

        try:
            print("🤖 Using AI to generate query...")
            response = self.model.generate_content(prompt)

            if response and response.text:
                sql = response.text.strip()

                # Remove markdown code blocks if present
                if sql.startswith("```sql"):
                    sql = sql[6:]
                if sql.startswith("```"):
                    sql = sql[3:]
                if sql.endswith("```"):
                    sql = sql[:-3]

                sql = sql.strip()

                # Check for error messages
                if sql.startswith("❌"):
                    print(sql)
                    print(
                        "💡 Try asking: 'what datasets are available?' or 'what tables are in [dataset]?'"
                    )
                    return None

                print(
                    f"🤔 AI Thinking: {response.text[:200]}..."
                    if len(response.text) > 200
                    else f"🤔 AI Thinking: {response.text}"
                )
                print(f"📝 Generated SQL:\n   {sql}")

                return sql

        except Exception as e:
            print(f"❌ AI query generation failed: {e}")

        return None

    def get_ai_error_suggestions(
        self, sql: str, error: str, original_question: str
    ) -> Optional[str]:
        """Use AI to analyze query errors and suggest fixes"""
        if not self.model:
            return None

        try:
            # Get available datasets context
            datasets_context = ""
            if self.metadata_cache and self.metadata_cache.datasets_cache:
                datasets = self.metadata_cache.datasets_cache.get("datasets", [])
                datasets_context = f"Available datasets: {', '.join(datasets)}"

            # Create a focused prompt for error analysis
            error_prompt = f"""
            A BigQuery SQL query failed with an error. Please analyze the error and suggest specific fixes.

            Original Question: {original_question}

            Generated SQL:
            {sql}

            Error Message:
            {error}

            Available Context:
            - Project: {self.project_id}
            - {datasets_context}

            Please provide:
            1. A brief explanation of what caused the error
            2. 1-2 specific suggestions to fix the query
            3. If possible, a corrected version of the SQL

            Keep the response concise and actionable.
            """

            print("🤖 Asking AI for error analysis...")

            response = self.model.generate_content(error_prompt)
            suggestion = response.text.strip()

            return suggestion

        except Exception as e:
            print(f"⚠️  Could not get AI suggestions: {e}")
            return None

    def _build_schema_context(self, schemas: Dict[str, Dict]) -> str:
        """Build formatted schema context for the prompt"""
        if not schemas:
            return ""

        context_parts = ["TABLE SCHEMAS:"]

        for table_name, schema in schemas.items():
            columns_info = []
            for col in schema["columns"]:
                col_desc = f"  - {col['name']} ({col['type']}, {col['mode']})"
                if col["description"]:
                    col_desc += f" - {col['description']}"
                columns_info.append(col_desc)

            table_info = f"""
Table: {table_name}
Columns:
{chr(10).join(columns_info)}
Rows: {schema.get("num_rows", "Unknown")}
"""
            context_parts.append(table_info)

        return "\n".join(context_parts)
