# BigQuery Agent with AI-Powered Query Generation

This workspace contains a streamlined BigQuery agent that generates and executes SQL queries from natural language questions using Google Gemini AI.

## ✨ Features

- **🤖 AI-Powered Only**: Uses Google Gemini to generate smart SQL queries (required)
- **📊 Instant Results**: Executes queries immediately without confirmation dialogs  
- **💾 Auto-Save**: Saves all results as JSON files with date-ordered names
- **🧠 Thinking Steps**: Shows AI reasoning before generating queries
- **🔒 Cost-Safe**: Includes safety limits and smart query patterns
- **� Bytes Estimation**: Shows data processed and cost before running queries
- **�📁 Organized**: Date-ordered filenames for easy chronological tracking
- **🚀 Enhanced Discovery**: Built-in metadata commands for easy exploration
- **📋 Smart Context**: AI gets table/column information for better query generation
- **🔧 AI Error Analysis**: When queries fail, AI analyzes errors and suggests fixes
- **📖 Table Descriptions**: Get human-readable explanations of what tables contain
- **⚡ Smart Caching**: Persistent metadata cache for faster responses

## 🚀 Quick Start

```bash
python3 bigquery_ai_gen_exec.py "datasets"
```

## 🛠️ Requirements

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or manually:
pip install google-cloud-bigquery google-generativeai
```

### 2. Authentication (Required)
```bash
# Google Cloud authentication
gcloud auth application-default login

# Get Gemini API key from https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="your-api-key-here"
```

**Note**: Both BigQuery access and Gemini API key are required. If either is missing, the tool will exit with an error.

### 3. Run
```bash
# Single question
python3 bigquery_ai_gen_exec.py "your question here"

# Interactive mode  
python3 bigquery_ai_gen_exec.py
```

## 🆘 Discovery Commands

**Quick Commands (no AI needed):**
- `datasets` - List all available datasets
- `explore <dataset>` - Show tables in a dataset  
- `schema <dataset.table>` - Show column details for a table
- `describe <dataset.table>` - Get AI analysis of what the table contains
- `help` - Show all available commands

**Cache Management:**
- `cache-info` - Show cache status, size, and contents
- `clear-cache` - Clear all cached metadata
- `clear-descriptions` - Clear only cached table descriptions

## 📝 Example Usage

### Start with Discovery
```bash
# See what's available
python3 bigquery_ai_gen_exec.py "datasets"

# Explore a specific dataset
python3 bigquery_ai_gen_exec.py "explore bi"

# Check table schema
python3 bigquery_ai_gen_exec.py "schema bi.my_table"
```

### AI-Powered Queries
```bash
# Discovery queries (AI-powered but use INFORMATION_SCHEMA)
python3 bigquery_ai_gen_exec.py "what tables are in the bi dataset?"

# Data analysis (specify dataset.table clearly)
python3 bigquery_ai_gen_exec.py "show me 10 sample records from bi.my_table"
python3 bigquery_ai_gen_exec.py "how many rows are in bi.my_table?"

# Complex analysis
python3 bigquery_ai_gen_exec.py "what are the top 5 most common values in host_arch from bi.profiling_data?"
```

## 🎯 How It Works

1. **Ask Question**: You provide a natural language question or use discovery commands
2. **Metadata Context**: The system provides table/column context to the AI when relevant
3. **AI Analysis**: Gemini AI analyzes the question and shows thinking steps  
4. **Query Generation**: AI generates optimized BigQuery SQL (with metadata awareness)
5. **Bytes Estimation**: System estimates data to be processed and shows cost before execution
6. **Confirmation**: For large queries (>100MB), asks for confirmation in interactive mode
7. **Execution**: Query runs with real-time progress and estimation accuracy
8. **Results Display**: Results shown in console with summary
9. **Auto-Save**: Full results saved as JSON with date-ordered filename

## 🔥 Advanced Features

### Cost Estimation & Protection
Before executing any query, the system estimates the data that will be processed:

```
🔍 Estimating query cost...
📊 Query will process: 45.2 MB
💰 Estimated cost: $0.0002 USD

⚡ Executing query...
✅ Query executed successfully!
   Rows returned: 1,247
   Bytes processed: 47.1 MB
   Estimation accuracy: 95.8%
```

**For large queries (>100MB or >$0.01):**
```
⚠️  This query will process a significant amount of data!
   Data to process: 2.3 GB
   Estimated cost: $0.0115 USD
   Continue? (y/N):
```

### AI Error Analysis & Recovery
When queries fail, the AI automatically analyzes the error and provides helpful suggestions:

```
❌ Query failed: Syntax error: Unexpected keyword WHERE at [1:15]

🔍 Analyzing error with AI...

💡 AI Suggestions:
1. Error Cause: The SQL query is missing the FROM clause's table specification
2. Fixes: Specify the table name after FROM  
3. Corrected SQL: SELECT * FROM `project.dataset.table` WHERE column = 'value'
```

### Table Descriptions with AI
Use `describe <dataset.table>` to get human-readable analysis:

```bash
python3 bigquery_ai_gen_exec.py "describe bi.profiling_data"
```

**Example Output:**
```
📊 Table Analysis:
🔢 Size: 11,176 rows, 74 columns
🎯 Purpose: Appears to be a log/event tracking table
📋 Data Structure:
   🔢 Metrics/Numbers: ess_logs, ess_logs_GB, clusters_ingest_logs
   📝 Text data: cluster_name, real_cluster_name
🔍 Sample insights:
   • cluster_name: dd956fef7fee47b1a2bb8d5b3c1d830b
```

### Smart Metadata Caching
- **Persistent cache**: Stored in `metadata_cache.json`, survives restarts
- **24-hour expiration**: Automatically refreshes when needed
- **Performance boost**: Eliminates repeated metadata queries
- **Cache management**: Use `cache-info` to see status, `clear-cache` to reset

## 📁 Project Structure

```
bigquery_gen_executor/
├── bigquery_ai_gen_exec.py    # Main BigQuery agent (⭐ YOUR MAIN FILE)
├── metadata_cache.py           # Metadata caching logic
├── ai_query_generator.py       # AI query generation logic
├── metadata_cache.json         # Persistent cache (auto-generated)
├── requirements.txt            # Python dependencies
├── query_results/              # Auto-saved JSON results
│   ├── 2025-07-30_17-25-48_what_tables_are_in_the_bi_dataset.json
│   └── 2025-07-30_17-26-19_show_me_the_top_3_elasticsearch_clusters.json
├── CACHING_GUIDE.md           # Detailed cache management guide
└── README.md                  # This file
```

## 🔧 Architecture & Performance

### Modular Design
- **Separated concerns**: Main script imports `MetadataCache` and `AIQueryGenerator` classes
- **Clean code structure**: Each module has a single responsibility  
- **Easy maintenance**: Bug fixes and features can be added to specific modules

### Intelligent Caching
- **Fast responses**: Metadata cached for 24 hours, dramatically reducing query times
- **Smart context**: AI gets relevant table/column information for better SQL generation
- **Automatic refresh**: Cache expires and refreshes automatically when needed

### Error Recovery
- **AI-powered analysis**: Understands BigQuery-specific errors and suggests fixes
- **Contextual help**: Suggestions based on your available datasets and tables
- **Learning system**: Gets better at helping with common issues over time

## 💡 Tips for Best Results

1. **Start with Discovery**: Use `datasets` → `explore <dataset>` → `schema <dataset.table>`
2. **Be Specific**: For data queries, always specify `dataset.table`
3. **Use AI for Complex**: Let AI handle complex filtering, aggregations, and joins
4. **Combine Approaches**: Use discovery commands to explore, then AI for analysis

## 🛠️ Troubleshooting

**Missing Dependencies**: Install with `pip install -r requirements.txt`
**Authentication Issues**: Run `gcloud auth application-default login`
**No API Key**: Set `export GEMINI_API_KEY="your-key"`
**Permission Errors**: Ensure your Google Cloud account has BigQuery access
**Cache Issues**: See `CACHING_GUIDE.md` for detailed cache management help
**Slow Performance**: Use `cache-info` to check cache status, `clear-cache` to reset if needed

## 🔐 Security & Cost Protection

- **Bytes Estimation**: Every query shows estimated data processing and cost before execution
- **Confirmation Prompts**: Large queries (>100MB or >$0.01) require confirmation in interactive mode
- **Estimation Accuracy**: Shows how accurate the estimation was after execution
- **Safety LIMIT clauses**: All AI-generated queries include appropriate LIMIT clauses
- **Read-only Operations**: No DDL operations (CREATE, DROP, etc.) - only SELECT statements
- **Efficient Patterns**: Metadata queries use optimized INFORMATION_SCHEMA patterns
- **Smart Caching**: Results cached to minimize repeated metadata queries and costs

**Cost Thresholds:**
- Queries >100MB: Shows warning and asks for confirmation
- Queries >$0.01: Requires explicit user confirmation
- Command-line mode: Shows warnings but continues (for automation)
