#!/Users/marciatejedaaristei/workspace/bigquery_gen_executor/venv/bin/python
"""
BigQuery AI Generator & Executor

This script executes BigQuery queries with AI-powered query generation.
It uses Google Gemini AI to generate SQL queries from natural language questions
and the Google Cloud BigQuery client library to execute them.
and the Google Cloud BigQuery client library to execute them.

Requires:
- google-cloud-bigquery
- google-generativeai
- GEMINI_API_KEY or GOOGLE_API_KEY environment variable
"""

import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, Tuple

try:
    from google.cloud import bigquery

    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    print("❌ Google Cloud BigQuery library not installed.")
    print("💡 Install it with: pip install google-cloud-bigquery")

# Import Google Generative AI for AI-powered query generation (required)
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print(
        "❌ Google Generative AI library not available. This tool requires AI-powered query generation."
    )
    print("   Install with: pip install google-generativeai")

# Import our custom modules
from ai_query_generator import AIQueryGenerator
from metadata_cache import MetadataCache


class DirectBigQueryExecutor:
    def __init__(self, project_id: str = "elastic-observability"):
        self.project_id = project_id
        self.client = None
        self.metadata_cache = None
        self.ai_generator = None
        self.results_dir = "query_results"

        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"📁 Created results directory: {self.results_dir}")

        if BIGQUERY_AVAILABLE:
            try:
                # Initialize BigQuery client
                self.client = bigquery.Client(project=project_id)
                print(f"✅ Connected to BigQuery project: {project_id}")

                # Initialize metadata cache
                if BIGQUERY_AVAILABLE:
                    self.metadata_cache = MetadataCache(self.client, project_id)

                # Initialize AI generator with metadata cache
                if GEMINI_AVAILABLE:
                    self.ai_generator = AIQueryGenerator(
                        project_id, self.metadata_cache
                    )

            except Exception as e:
                print(f"❌ Failed to connect to BigQuery: {e}")
                print("💡 Make sure you're authenticated with Google Cloud")
                print("   Run: gcloud auth application-default login")

    def handle_special_commands(self, question: str) -> Tuple[bool, str]:
        """Handle special discovery commands that don't require AI"""
        question_lower = question.lower().strip()

        # Check if metadata cache is available
        if not self.metadata_cache:
            if question_lower in [
                "datasets",
                "show datasets",
                "list datasets",
                "help",
                "commands",
                "?",
            ]:
                return (
                    True,
                    "❌ Metadata cache not available. Please ensure BigQuery is properly connected.",
                )
            return False, ""

        # Cache management commands
        if question_lower in [
            "clear-cache",
            "clear cache",
            "clean-cache",
            "clean cache",
        ]:
            return True, self.metadata_cache.clear_cache()

        if question_lower in [
            "clear-descriptions",
            "clear descriptions",
            "clean-descriptions",
            "clean descriptions",
        ]:
            return True, self.metadata_cache.clear_descriptions()

        if question_lower in [
            "cache-info",
            "cache info",
            "cache-status",
            "cache status",
        ]:
            return True, self.metadata_cache.get_cache_info()

        # Show datasets
        if question_lower in ["datasets", "show datasets", "list datasets"]:
            return True, self.metadata_cache.get_datasets_summary()

        # Explore dataset (show tables)
        if question_lower.startswith("explore "):
            dataset_name = question_lower.replace("explore ", "").strip()
            return True, self.metadata_cache.get_tables_summary(dataset_name)

        # Show schema
        if question_lower.startswith("schema "):
            table_ref = question_lower.replace("schema ", "").strip()
            if "." in table_ref:
                dataset_name, table_name = table_ref.split(".", 1)
                return True, self.metadata_cache.get_columns_summary(
                    dataset_name, table_name
                )
            else:
                return (
                    True,
                    "❌ Please specify table as 'dataset.table' (e.g., 'schema bi.my_table')",
                )

        # Describe table data
        if question_lower.startswith("describe "):
            table_ref = question_lower.replace("describe ", "").strip()
            if "." in table_ref:
                dataset_name, table_name = table_ref.split(".", 1)
                return True, self.describe_table_data(dataset_name, table_name)
            else:
                return (
                    True,
                    "❌ Please specify table as 'dataset.table' (e.g., 'describe bi.my_table')",
                )

        # Show thresholds info
        if question_lower in ["thresholds", "limits", "confirmation-settings"]:
            info = f"""⚙️  Query Execution Thresholds:
   📊 Warning threshold: 100 MB / $0.01 USD (shows warning)
   🚨 Force confirmation: 500 MB / $0.025 USD (requires confirmation even in CLI mode)
   🔄 Mode: {"Interactive" if len(sys.argv) == 1 else "Command-line"}
   
   Behavior:
   • Small queries (<100MB): Execute automatically
   • Medium queries (100MB-500MB): Warn in CLI mode, confirm in interactive mode  
   • Large queries (>500MB): Always require confirmation"""
            return True, info

        # Quick help
        if question_lower in ["help", "commands", "?"]:
            help_text = """🆘 Available Commands:

📊 Discovery Commands:
   • datasets              - List all available datasets
   • explore <dataset>     - Show tables in a dataset  
   • schema <dataset.table> - Show column details for a table
   • describe <dataset.table> - Get human-readable description of table data

🧹 Cache Management:
   • cache-info            - Show cache status and statistics
   • clear-cache           - Clear all cached metadata
   • clear-descriptions    - Clear only cached table descriptions

🤖 AI-Powered Queries (just ask naturally):
   • "what tables are in the bi dataset?"
   • "show me 5 samples from bi.my_table"
   • "how many rows are in bi.my_table?"
   • "what are the distinct values in column_name from dataset.table?"

💡 Tips:
   • Use 'datasets' to start exploring
   • Specify dataset.table for data queries
   • All results are auto-saved to query_results/
   • Cache expires automatically after 24 hours
   • If queries fail, AI will analyze errors and suggest fixes
"""
            return True, help_text

        return False, ""

    def generate_query(
        self,
        question: str,
    ) -> Tuple[str, str]:
        """Generate SQL query from natural language question using AI"""

        # Only use AI generation - no fallback
        if not self.ai_generator or not self.ai_generator.model:
            error_msg = "AI query generation is not available. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            return None, error_msg

        print("🔍 Analyzing question for table references...")

        # Extract and fetch table schemas
        table_refs = extract_table_references(question, self.client.project)
        schemas = {}

        if table_refs:
            print(f"🔍 Found {len(table_refs)} table reference(s) in user input")

            for project_id, dataset_id, table_id in table_refs:
                full_table_name = f"{project_id}.{dataset_id}.{table_id}"
                schema = get_table_schema_from_existing_command(
                    project_id, dataset_id, table_id, self.client, self.metadata_cache
                )
                if schema:
                    schemas[full_table_name] = schema

        if schemas:
            print(f"✅ Using schema context for {len(schemas)} table(s)")
            schema_context = build_schema_context(schemas)
            enhanced_question = (
                f"Using the following table schemas:\n{schema_context}\n\n"
                f"Original question: {question}"
            )
            return self.ai_generator.generate_sql_with_ai(
                enhanced_question, schemas
            ), None
        else:
            return self.ai_generator.generate_sql_with_ai(question, schemas), None

    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query directly against BigQuery with bytes estimation"""
        if not self.client:
            return {
                "status": "error",
                "error": "BigQuery client not initialized",
                "sql": sql,
            }

        # First, estimate the bytes that will be processed
        print("🔍 Estimating query cost...")
        estimation = self.estimate_query_bytes(sql)

        if estimation["status"] == "success":
            print(
                f"📊 Query will process: {self.format_bytes(estimation['estimated_bytes'])}"
            )
            print(f"💰 Estimated cost: ${estimation['estimated_cost_usd']:.4f} USD")

            # Check if confirmation is needed for large queries
            if self.should_confirm_execution(estimation):
                print("⚠️  This query will process a significant amount of data!")
                print(
                    f"   Data to process: {self.format_bytes(estimation['estimated_bytes'])}"
                )
                print(f"   Estimated cost: ${estimation['estimated_cost_usd']:.4f} USD")

                # Check if this needs confirmation even in command-line mode
                force_confirm = self.should_force_confirm_execution(estimation)

                # In interactive mode OR for very expensive queries, ask for confirmation
                if (
                    len(sys.argv) == 1 or force_confirm
                ):  # Interactive mode OR force confirm
                    try:
                        mode_msg = (
                            "(very expensive query)"
                            if force_confirm and len(sys.argv) > 1
                            else ""
                        )
                        confirm = (
                            input(f"   Continue? (y/N) {mode_msg}: ").strip().lower()
                        )
                        if confirm not in ["y", "yes"]:
                            print("❌ Query execution cancelled by user")
                            return {
                                "status": "cancelled",
                                "sql": sql,
                                "estimation": estimation,
                                "timestamp": datetime.now().isoformat(),
                            }
                    except KeyboardInterrupt:
                        print("\n❌ Query execution cancelled by user")
                        return {
                            "status": "cancelled",
                            "sql": sql,
                            "estimation": estimation,
                            "timestamp": datetime.now().isoformat(),
                        }
                else:
                    # In command-line mode for moderately expensive queries, show warning but continue
                    print("   ⚠️  Continuing with large query (command-line mode)")
        else:
            print(
                f"⚠️  Could not estimate query cost: {estimation.get('error', 'Unknown error')}"
            )
            print("   Proceeding with execution...")

        print()
        print("⚡ Executing query:")
        print(f"   {sql}")
        print()

        try:
            # Execute the query
            query_job = self.client.query(sql)
            results = query_job.result()

            # Convert results to list of dictionaries
            rows = []
            for row in results:
                row_dict = {}
                for key, value in row.items():
                    row_dict[key] = value
                rows.append(row_dict)

            # Prepare response
            response = {
                "status": "success",
                "sql": sql,
                "data": rows,
                "row_count": len(rows),
                "bytes_processed": query_job.total_bytes_processed,
                "bytes_estimated": estimation.get("estimated_bytes")
                if estimation["status"] == "success"
                else None,
                "timestamp": datetime.now().isoformat(),
            }

            print("✅ Query executed successfully!")
            print(f"   Rows returned: {len(rows)}")
            print(
                f"   Bytes processed: {self.format_bytes(query_job.total_bytes_processed)}"
            )

            # Show estimation accuracy if available
            if estimation["status"] == "success":
                estimated = estimation["estimated_bytes"]
                actual = query_job.total_bytes_processed
                if estimated > 0:
                    accuracy = (actual / estimated) * 100
                    print(f"   Estimation accuracy: {accuracy:.1f}%")

            return response

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Query failed: {error_msg}")

            # Create the basic error response
            error_response = {
                "status": "error",
                "sql": sql,
                "error": error_msg,
                "estimation": estimation if estimation["status"] == "success" else None,
                "timestamp": datetime.now().isoformat(),
            }

            return error_response

    def save_results_to_file(
        self, result: Dict[str, Any], format: str = "json", filename: str = None
    ) -> str:
        """Save query results to a file in various formats"""
        if result["status"] == "error":
            print("❌ Cannot save error results to file")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not filename:
            filename = f"bigquery_result_{timestamp}"

        if format.lower() == "json":
            filepath = os.path.join(self.results_dir, f"{filename}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)

        elif format.lower() == "csv":
            filepath = os.path.join(self.results_dir, f"{filename}.csv")
            data = result["data"]

            if not data:
                print("❌ No data to save to CSV")
                return None

            # Get all unique keys from all rows
            fieldnames = set()
            for row in data:
                if isinstance(row, dict):
                    fieldnames.update(row.keys())
            fieldnames = sorted(list(fieldnames))

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    if isinstance(row, dict):
                        writer.writerow(row)

        elif format.lower() == "txt":
            filepath = os.path.join(self.results_dir, f"{filename}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("BigQuery Query Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Question: {result.get('question', 'N/A')}\n")
                f.write(f"SQL: {result['sql']}\n")
                f.write(f"Timestamp: {result['timestamp']}\n")
                f.write(f"Rows: {result['row_count']}\n")
                f.write(f"Bytes Processed: {result.get('bytes_processed', 'N/A')}\n")
                f.write("\n" + "=" * 50 + "\n\n")

                data = result["data"]
                if not data:
                    f.write("No data returned\n")
                elif len(data) == 1 and len(data[0]) == 1:
                    # Single value result
                    key, value = list(data[0].items())[0]
                    f.write(f"{key}: {value}\n")
                else:
                    # Multiple rows
                    for i, row in enumerate(data, 1):
                        f.write(f"Row {i}:\n")
                        if isinstance(row, dict):
                            for key, value in row.items():
                                f.write(f"  {key}: {value}\n")
                        else:
                            f.write(f"  {row}\n")
                        f.write("\n")
        else:
            print(f"❌ Unsupported format: {format}")
            return None

        return filepath

    def display_results(self, result: Dict[str, Any], save_to_file: bool = True):
        """Display query results in a user-friendly format"""
        if result["status"] == "error":
            print(f"❌ Error: {result['error']}")
            return

        # Handle metadata commands (don't save these to files)
        if result.get("is_metadata_command"):
            return  # Already printed in ask_and_execute

        data = result["data"]
        print(f"\n📊 Results ({result['row_count']} rows):")
        print("=" * 60)

        if not data:
            print("   No data returned")
            return

        # Always save to file by default (JSON only)
        saved_files = []
        if save_to_file:
            # Use ISO format for better date ordering: YYYY-MM-DD_HH-MM-SS
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Create a meaningful filename from the question
            question = result.get("question", "query")
            # Clean the question for filename use
            safe_question = "".join(
                c for c in question if c.isalnum() or c in " -_"
            ).rstrip()
            safe_question = safe_question.replace(" ", "_").lower()[
                :40
            ]  # Increased length

            # Format: YYYY-MM-DD_HH-MM-SS_question_description
            filename_base = (
                f"{timestamp}_{safe_question}"
                if safe_question
                else f"{timestamp}_query"
            )

            # Save as JSON only
            json_file = self.save_results_to_file(result, "json", filename_base)
            if json_file:
                saved_files.append(f"JSON: {json_file}")
                print(f"\n💾 Results saved to: {json_file}")

        # Display summary in console
        if len(data) == 1 and len(data[0]) == 1:
            # Single value result (like COUNT)
            key, value = list(data[0].items())[0]
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:,}")
            else:
                print(f"   {key}: {value}")

        elif len(data) <= 5:
            # Show all rows for small results
            for i, row in enumerate(data, 1):
                print(f"\n   Row {i}:")
                for key, value in row.items():
                    display_value = (
                        f"{value:,}" if isinstance(value, (int, float)) else str(value)
                    )
                    # Truncate long strings
                    if len(display_value) > 50:
                        display_value = display_value[:47] + "..."
                    print(f"     {key}: {display_value}")

        else:
            # Show first few rows for large results
            print("   First 3 rows:")
            for i, row in enumerate(data[:3], 1):
                print(f"\n   Row {i}:")
                for key, value in row.items():
                    display_value = (
                        f"{value:,}" if isinstance(value, (int, float)) else str(value)
                    )
                    if len(display_value) > 50:
                        display_value = display_value[:47] + "..."
                    print(f"     {key}: {display_value}")

            print(f"\n   ... and {len(data) - 3} more rows")

        return saved_files

    def get_ai_error_suggestions(
        self, sql: str, error: str, original_question: str
    ) -> str:
        """Use AI to analyze query errors and suggest fixes"""
        if not self.ai_generator or not self.ai_generator.model:
            return None

        try:
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
            - Available datasets: {", ".join(self.metadata_cache.datasets_cache.keys()) if self.metadata_cache else "Unknown"}

            Please provide:
            1. A brief explanation of what caused the error
            2. 1-2 specific suggestions to fix the query
            3. If possible, a corrected version of the SQL

            Keep the response concise and actionable.
            """

            print("🤖 Asking AI for error analysis...")

            response = self.ai_generator.model.generate_content(error_prompt)
            suggestion = response.text.strip()

            return suggestion

        except Exception as e:
            print(f"⚠️  Could not get AI suggestions: {e}")
            return None

    def clean_sql_query(self, query: str) -> str:
        """Clean SQL query by removing markdown formatting and extra whitespace"""
        # Handle None input
        if query is None:
            return ""

        # Remove markdown code blocks
        query = re.sub(r"```sql\s*", "", query)
        query = re.sub(r"```\s*", "", query)

        # Remove leading/trailing whitespace
        query = query.strip()

        # Remove any extra newlines or spaces
        query = " ".join(query.split())

        return query

    def generate_query_with_schema(self, user_question: str, schemas: dict) -> str:
        """Generate SQL query with table schema context"""
        schema_context = self.build_schema_context(schemas)

        prompt = f"""Given the following table schemas and user request, generate a BigQuery SQL query.

TABLE SCHEMAS:
{schema_context}

USER REQUEST:
{user_question}

Please generate a valid BigQuery SQL query that:
1. Uses the correct column names from the provided schemas
2. Uses appropriate data types and functions
3. Follows BigQuery SQL syntax
4. Is optimized for performance

Return ONLY the SQL query without any explanations, markdown formatting, or code blocks.

SQL Query:"""

        print("🤖 Using schema-aware AI to generate query...")
        response = self.ai_generator.model.generate_content(prompt)
        sql_query = response.text.strip()

        return sql_query

    def describe_table_data(self, dataset_name: str, table_name: str) -> str:
        """Get AI-powered description of table data"""
        if not self.ai_generator or not self.ai_generator.model:
            return "❌ AI not available for table descriptions"

        return self.metadata_cache.get_table_description(dataset_name, table_name)

    def estimate_query_bytes(self, sql: str) -> Dict[str, Any]:
        """Estimate bytes that will be processed by a query"""
        try:
            # Create a dry run job
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            query_job = self.client.query(sql, job_config=job_config)

            estimated_bytes = query_job.total_bytes_processed
            # BigQuery pricing: $5 per TB
            estimated_cost_usd = (estimated_bytes / (1024**4)) * 5

            return {
                "status": "success",
                "estimated_bytes": estimated_bytes,
                "estimated_cost_usd": estimated_cost_usd,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "estimated_bytes": 0,
                "estimated_cost_usd": 0,
            }

    def format_bytes(self, bytes_value: int) -> str:
        """Format bytes in human readable format"""
        if bytes_value == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB", "TB"]
        size = bytes_value
        unit_index = 0

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        return f"{size:.1f} {units[unit_index]}"

    def should_confirm_execution(self, estimation: Dict[str, Any]) -> bool:
        """Check if query execution should be confirmed based on size/cost"""
        if estimation["status"] != "success":
            return False

        # Threshold: 100MB or $0.01
        return (
            estimation["estimated_bytes"] > 100 * 1024 * 1024
            or estimation["estimated_cost_usd"] > 0.01
        )

    def should_force_confirm_execution(self, estimation: Dict[str, Any]) -> bool:
        """Check if query should force confirmation even in CLI mode"""
        if estimation["status"] != "success":
            return False

        # Force confirmation for very expensive queries: 500MB or $0.025
        return (
            estimation["estimated_bytes"] > 500 * 1024 * 1024
            or estimation["estimated_cost_usd"] > 0.025
        )

    def ask_and_execute(self, question: str) -> Dict[str, Any]:
        """Main method to process a question and execute the appropriate action"""
        # Handle special commands first
        is_special, response = self.handle_special_commands(question)
        if is_special:
            print(response)
            return {
                "status": "success",
                "question": question,
                "response": response,
                "is_metadata_command": True,
                "timestamp": datetime.now().isoformat(),
            }

        # Generate SQL query using AI
        sql_query, error = self.generate_query(question)

        if error:
            return {
                "status": "error",
                "question": question,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }

        # Clean the generated SQL
        sql_query = self.clean_sql_query(sql_query)

        # Execute the query
        result = self.execute_query(sql_query)
        result["question"] = question

        # If query failed, try to get AI suggestions
        if result["status"] == "error":
            print()
            print("🔍 Analyzing error with AI...")
            suggestions = self.get_ai_error_suggestions(
                sql_query, result["error"], question
            )

            if suggestions:
                print()
                print("💡 AI Suggestions:")
                print("=" * 50)
                print(suggestions)
                print("=" * 50)
                result["ai_suggestions"] = suggestions

        return result


def extract_table_references(user_input: str, default_project: str) -> list:
    """Extract table references from user input"""
    import re

    pattern = r"(?:([a-zA-Z0-9_-]+)\.)?([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_-]+)"
    matches = re.findall(pattern, user_input)

    tables = []
    for match in matches:
        project_id = match[0] or default_project
        dataset_id = match[1]
        table_id = match[2]
        tables.append((project_id, dataset_id, table_id))

    return tables


def get_table_schema_from_existing_command(
    project_id: str, dataset_id: str, table_id: str, client, cache
):
    """Use the existing schema command functionality to get table schema"""
    # First check if we have it cached
    cached_schema = cache.get_table_schema(project_id, dataset_id, table_id)
    if cached_schema:
        print(f"✅ Found cached schema for {project_id}.{dataset_id}.{table_id}")
        return cached_schema

    # Fetch from BigQuery
    try:
        table = client.get_table(f"{project_id}.{dataset_id}.{table_id}")

        schema_info = {
            "table_id": f"{project_id}.{dataset_id}.{table_id}",
            "columns": [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or "",
                }
                for field in table.schema
            ],
            "num_rows": table.num_rows,
            "size_bytes": table.num_bytes,
        }

        print(f"📋 Available columns in {project_id}.{dataset_id}.{table_id}:")
        for col in schema_info["columns"]:
            print(f"  - {col['name']} ({col['type']})")

        # Save to cache
        cache.save_table_schema(project_id, dataset_id, table_id, schema_info)
        print(f"💾 Schema cached for {project_id}.{dataset_id}.{table_id}")

        return schema_info

    except Exception as e:
        print(f"❌ Error fetching schema: {e}")
        return None


def build_schema_context(schemas: dict) -> str:
    """Build formatted schema context for the prompt"""
    if not schemas:
        return "No table schemas available."

    context_parts = []
    for table_name, schema in schemas.items():
        columns_info = []
        for col in schema["columns"]:
            col_desc = f"  - {col['name']} ({col['type']}, {col['mode']})"
            if col["description"]:
                col_desc += f" - {col['description']}"
            columns_info.append(col_desc)

        table_info = f"""Table: {table_name}
Columns:
{chr(10).join(columns_info)}
Rows: {schema.get("num_rows", "Unknown")}"""
        context_parts.append(table_info)

    return "\n\n".join(context_parts)


def main():
    """Main entry point"""
    if not BIGQUERY_AVAILABLE:
        print("\n💡 To install the required library:")
        print("   pip install google-cloud-bigquery")
        print("\n💡 To authenticate:")
        print("   gcloud auth application-default login")
        return

    if not GEMINI_AVAILABLE:
        print("\n❌ This tool requires Google Generative AI for query generation.")
        print("💡 To install:")
        print("   pip install google-generativeai")
        print("💡 Set your API key:")
        print("   export GEMINI_API_KEY=your_api_key_here")
        return

    executor = DirectBigQueryExecutor()

    # Check if AI is properly configured
    if not executor.ai_generator or not executor.ai_generator.model:
        print("❌ AI query generation not configured.")
        print("💡 Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        return

    print("✅ AI-powered query generation: ENABLED (Google Gemini)")

    if len(sys.argv) > 1:
        # Command line question
        question = " ".join(sys.argv[1:])
        result = executor.ask_and_execute(question)
        executor.display_results(result)

        # Final summary for command-line usage
        if result["status"] == "success":
            print("\n✅ Query completed successfully!")
            print(
                f"   Results automatically saved to {executor.results_dir}/ directory"
            )
    else:
        # Interactive mode
        print("🚀 Direct BigQuery Executor with AI (Google Gemini)")
        print("=" * 55)
        print("Ask questions and get immediate results saved to files!")
        print("No confirmation dialogs, no intermediary steps.")
        print(f"All results automatically saved to: {executor.results_dir}/")
        print()
        print("🆘 Quick Discovery Commands:")
        print("• datasets                    - Show all available datasets")
        print("• explore <dataset>           - Show tables in a dataset")
        print("• schema <dataset.table>      - Show column details")
        print("• describe <dataset.table>    - Understand what data a table contains")
        print("• help                        - Show all commands")
        print()
        print("🤖 AI can handle complex questions like:")
        print("• What are the top 5 host architectures by count?")
        print("• Show me records from the last hour")
        print("• Which Elasticsearch clusters have the most data?")
        print("• Find hosts with more than 8 CPU cores")
        print()
        print("📋 Basic questions work too:")
        print("• How many rows are there?")
        print("• Show me 5 sample records")
        print("• What columns are available?")
        print()
        print("💡 Start with 'datasets' to explore your data!")
        print()

        while True:
            try:
                question = input("❓ Your question (or 'quit'): ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if question:
                    result = executor.ask_and_execute(question)
                    executor.display_results(result)

                    # Show quick summary
                    if result["status"] == "success":
                        print("✅ Query completed - results saved to files above!")
                    print()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
