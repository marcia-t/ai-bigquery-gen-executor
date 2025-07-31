"""
BigQuery Metadata Cache Module

Handles caching of BigQuery schema metadata including datasets, tables, and columns.
Provides persistent file-based caching with expiration support.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict


class MetadataCache:
    """Cache for BigQuery schema metadata to improve user experience"""

    def __init__(self, client, project_id: str):
        self.client = client
        self.project_id = project_id
        self.datasets_cache = {}
        self.tables_cache = {}
        self.columns_cache = {}
        self.cache_timestamp = None
        self.cache_file = "metadata_cache.json"
        self.cache_expiry_hours = 24  # Cache expires after 24 hours

        # Load existing cache if available
        self.load_cache_from_file()

    def load_cache_from_file(self):
        """Load cache from JSON file if it exists and is not expired"""
        try:
            if not os.path.exists(self.cache_file):
                return

            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)

            # Check if cache is expired - try both field names for compatibility
            cache_timestamp = cache_data.get("cache_timestamp") or cache_data.get("timestamp")
            if cache_timestamp:
                cache_time = datetime.fromisoformat(cache_timestamp)
                if datetime.now() - cache_time > timedelta(hours=self.cache_expiry_hours):
                    print("📋 Cache expired, will refresh on next use")
                    return

            # Load cache data
            self.datasets_cache = cache_data.get("datasets_cache", {})
            self.tables_cache = cache_data.get("tables_cache", {})
            self.columns_cache = cache_data.get("columns_cache", {})
            self.cache_timestamp = cache_timestamp

            # Count items for display
            datasets_count = len(self.datasets_cache.get("datasets", []))
            tables_count = sum(
                len(tc.get("tables", [])) for tc in self.tables_cache.values()
            )

            print(f"📋 Loaded cached metadata: {datasets_count} datasets, {tables_count} tables")

        except Exception as e:
            print(f"⚠️  Could not load cache: {e}")

    def save_cache_to_file(self):
        """Save current cache to JSON file"""
        try:
            cache_data = {
                "cache_timestamp": datetime.now().isoformat(),
                "project_id": self.project_id,
                "datasets_cache": self.datasets_cache,
                "tables_cache": self.tables_cache,
                "columns_cache": self.columns_cache,
            }

            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)

            # Count items for display
            datasets_count = len(self.datasets_cache.get("datasets", []))
            tables_count = sum(
                len(tc.get("tables", [])) for tc in self.tables_cache.values()
            )

            print(f"💾 Saved metadata cache: {datasets_count} datasets, {tables_count} tables")

        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")

    def refresh_datasets(self) -> Dict[str, Any]:
        """Refresh and cache dataset information"""
        try:
            datasets = []
            for dataset in self.client.list_datasets():
                datasets.append(dataset.dataset_id)

            self.datasets_cache = {
                "datasets": datasets,
                "count": len(datasets),
                "timestamp": datetime.now().isoformat(),
            }
            print(f"📋 Cached {len(datasets)} datasets")

            # Save to file after refresh
            self.save_cache_to_file()

            return self.datasets_cache
        except Exception as e:
            print(f"❌ Failed to cache datasets: {e}")
            return {}

    def refresh_tables_for_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Refresh and cache table information for a specific dataset"""
        try:
            dataset_ref = self.client.dataset(dataset_name)
            tables = []

            for table in self.client.list_tables(dataset_ref):
                tables.append(
                    {
                        "name": table.table_id,
                        "type": table.table_type,
                        "created": table.created.isoformat() if table.created else None,
                    }
                )

            self.tables_cache[dataset_name] = {
                "tables": tables,
                "count": len(tables),
                "timestamp": datetime.now().isoformat(),
            }
            print(f"📋 Cached {len(tables)} tables for dataset '{dataset_name}'")

            # Save to file after refresh
            self.save_cache_to_file()

            return self.tables_cache[dataset_name]
        except Exception as e:
            print(f"❌ Failed to cache tables for {dataset_name}: {e}")
            return {}

    def refresh_columns_for_table(
        self, dataset_name: str, table_name: str
    ) -> Dict[str, Any]:
        """Refresh and cache column information for a specific table"""
        try:
            query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM `{self.project_id}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            results = self.client.query(query).result()

            columns = []
            for row in results:
                columns.append(
                    {
                        "name": row.column_name,
                        "type": row.data_type,
                        "nullable": row.is_nullable == "YES",
                        "default": row.column_default,
                        "description": None,  # Not available in all schemas
                    }
                )

            table_key = f"{dataset_name}.{table_name}"
            self.columns_cache[table_key] = {
                "columns": columns,
                "count": len(columns),
                "timestamp": datetime.now().isoformat(),
            }
            print(f"📋 Cached {len(columns)} columns for table '{table_key}'")

            # Save to file after refresh
            self.save_cache_to_file()

            return self.columns_cache[table_key]
        except Exception as e:
            print(f"❌ Failed to cache columns for {dataset_name}.{table_name}: {e}")
            return {}

    def get_datasets_summary(self) -> str:
        """Get a formatted summary of available datasets"""
        if not self.datasets_cache:
            self.refresh_datasets()

        if not self.datasets_cache:
            return "No datasets available or accessible."

        datasets = self.datasets_cache["datasets"]
        summary = f"📊 Available Datasets ({len(datasets)}):\n"
        for dataset in datasets:
            table_count = len(self.tables_cache.get(dataset, {}).get("tables", []))
            if table_count > 0:
                summary += f"   • {dataset} ({table_count} tables)\n"
            else:
                summary += f"   • {dataset}\n"

        summary += "\n💡 Use: 'explore <dataset>' to see tables in a dataset"
        return summary

    def get_tables_summary(self, dataset_name: str) -> str:
        """Get a formatted summary of tables in a dataset"""
        if dataset_name not in self.tables_cache:
            self.refresh_tables_for_dataset(dataset_name)

        cache_data = self.tables_cache.get(dataset_name, {})
        tables = cache_data.get("tables", [])

        if not tables:
            return f"No tables found in dataset '{dataset_name}' or dataset doesn't exist."

        summary = f"📋 Tables in '{dataset_name}' ({len(tables)}):\n"
        for table in tables[:20]:  # Show first 20 tables
            summary += f"   • {table['name']} ({table['type']})\n"

        if len(tables) > 20:
            summary += f"   ... and {len(tables) - 20} more tables\n"

        summary += f"\n💡 Use: 'schema {dataset_name}.{tables[0]['name']}' to see column details"
        return summary

    def get_columns_summary(self, dataset_name: str, table_name: str) -> str:
        """Get a formatted summary of columns in a table"""
        table_key = f"{dataset_name}.{table_name}"

        if table_key not in self.columns_cache:
            self.refresh_columns_for_table(dataset_name, table_name)

        cache_data = self.columns_cache.get(table_key, {})
        columns = cache_data.get("columns", [])

        if not columns:
            return f"No columns found for table '{table_key}' or table doesn't exist."

        summary = f"🏗️  Schema for '{table_key}' ({len(columns)} columns):\n"
        for col in columns:
            summary += f"   • {col['name']}: {col['type']}\n"

        summary += f"\n💡 Use: 'describe {table_key}' to understand what data this table contains"
        return summary

    def get_context_for_ai(
        self, question: str, max_datasets: int = 5, max_tables_per_dataset: int = 10
    ) -> str:
        """Get relevant context for AI query generation"""
        context = f"Project: {self.project_id}\n\n"

        # Always include datasets
        if not self.datasets_cache:
            self.refresh_datasets()

        datasets = self.datasets_cache.get("datasets", [])[:max_datasets]
        context += f"Available datasets: {', '.join(datasets)}\n\n"

        # Check if question mentions specific datasets
        mentioned_datasets = [d for d in datasets if d.lower() in question.lower()]

        if mentioned_datasets:
            context += "Relevant tables:\n"
            for dataset in mentioned_datasets:
                if dataset not in self.tables_cache:
                    self.refresh_tables_for_dataset(dataset)

                tables = self.tables_cache.get(dataset, {}).get("tables", [])
                table_names = [t["name"] for t in tables[:max_tables_per_dataset]]
                context += f"  {dataset}: {', '.join(table_names)}\n"

        return context

    def clear_cache(self) -> str:
        """Clear all cached data"""
        try:
            # Clear in-memory cache
            self.datasets_cache = {}
            self.tables_cache = {}
            self.columns_cache = {}
            self.cache_timestamp = None

            # Remove cache file
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                return "🧹 Cache cleared successfully! All metadata will be refreshed on next use."
            else:
                return "🧹 Cache was already empty (no cache file found)."

        except Exception as e:
            return f"❌ Failed to clear cache: {e}"

    def clear_descriptions(self) -> str:
        """Clear only the cached descriptions while keeping schema data"""
        try:
            descriptions_cleared = 0
            
            # Remove descriptions from columns cache
            for table_key, table_data in self.columns_cache.items():
                if "description" in table_data:
                    del table_data["description"]
                    descriptions_cleared += 1
                if "description_timestamp" in table_data:
                    del table_data["description_timestamp"]
            
            # Save updated cache
            self.save_cache_to_file()
            
            if descriptions_cleared > 0:
                return f"🧹 Cleared {descriptions_cleared} cached table descriptions! Schema data remains cached."
            else:
                return "🧹 No cached descriptions found to clear."

        except Exception as e:
            return f"❌ Failed to clear descriptions: {e}"

    def get_cache_info(self) -> str:
        """Get information about current cache status"""
        try:
            if not os.path.exists(self.cache_file):
                return "📋 No cache file found - cache is empty."

            # Get file size
            file_size = os.path.getsize(self.cache_file)
            file_size_mb = file_size / (1024 * 1024)

            # Get cache age
            if self.cache_timestamp:
                cache_time = datetime.fromisoformat(self.cache_timestamp)
                age = datetime.now() - cache_time
                age_hours = age.total_seconds() / 3600

                # Check if expired
                expired = age_hours >= self.cache_expiry_hours
                expiry_status = "⚠️ EXPIRED" if expired else "✅ Valid"

                # Count cached items
                datasets_count = len(self.datasets_cache.get("datasets", []))
                tables_count = sum(
                    len(tc.get("tables", [])) for tc in self.tables_cache.values()
                )
                columns_count = len(self.columns_cache)
                
                # Count descriptions
                descriptions_count = sum(
                    1 for table_data in self.columns_cache.values()
                    if table_data.get("description")
                )

                info = f"""📋 Cache Information:
   📁 File: {self.cache_file}
   💾 Size: {file_size_mb:.2f} MB ({file_size:,} bytes)
   🕒 Age: {age_hours:.1f} hours ({expiry_status})
   📊 Cached items:
      • {datasets_count} datasets
      • {tables_count} tables across {len(self.tables_cache)} datasets
      • {columns_count} table schemas
      • {descriptions_count} table descriptions
   ⏰ Expires after: {self.cache_expiry_hours} hours

💡 Use 'clear-cache' to clean all cached data"""

                return info
            else:
                return "📋 Cache file exists but no timestamp found."

        except Exception as e:
            return f"❌ Failed to get cache info: {e}"
