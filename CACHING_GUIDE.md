# Metadata Caching Behavior

## 🔄 How Caching Works

### **Dataset Caching (`datasets` command)**

#### First Time:
```bash
python3 bigquery_ai_gen_exec.py "datasets"
```
**Output:**
- ✅ **Makes BigQuery API call** to `INFORMATION_SCHEMA.SCHEMATA`
- 📋 **Shows:** `Cached 5 datasets`
- 💾 **Shows:** `Saved metadata cache: 5 datasets, 0 tables`
- 💾 **Creates:** `metadata_cache.json` file

#### Subsequent Times:
```bash
python3 bigquery_ai_gen_exec.py "datasets"
```
**Output:**
- ❌ **No API call** - uses cached data
- 📋 **Shows:** `Loaded cached metadata: 5 datasets, 0 tables`
- ⚡ **Instant response** (much faster)

### **Table Caching (`explore <dataset>` command)**

#### First Time:
```bash
python3 bigquery_ai_gen_exec.py "explore bi"
```
**Output:**
- ✅ **Makes BigQuery API call** to `bi.INFORMATION_SCHEMA.TABLES`
- 📋 **Shows:** `Cached 84 tables for dataset 'bi'`
- 💾 **Shows:** `Saved metadata cache: 5 datasets, 84 tables`
- 💾 **Updates:** `metadata_cache.json` file

#### Subsequent Times:
```bash
python3 bigquery_ai_gen_exec.py "explore bi"
```
**Output:**
- ❌ **No API call** - uses cached data
- 📋 **Shows:** `Loaded cached metadata: 5 datasets, 84 tables`
- ⚡ **Instant response** (much faster)

### **Column Caching (`schema <dataset.table>` command)**

#### First Time:
```bash
python3 bigquery_ai_gen_exec.py "schema bi.my_table"
```
**Output:**
- ✅ **Makes BigQuery API call** to `bi.INFORMATION_SCHEMA.COLUMNS`
- 📋 **Shows:** `Cached X columns for table 'bi.my_table'`
- 💾 **Updates:** `metadata_cache.json` file

#### Subsequent Times:
```bash
python3 bigquery_ai_gen_exec.py "schema bi.my_table"
```
**Output:**
- ❌ **No API call** - uses cached data
- ⚡ **Instant response** (much faster)

## 📁 Cache Storage

### **File Location:**
- **File:** `metadata_cache.json` (in same directory as script)
- **Format:** JSON with timestamps
- **Persistence:** Survives script restarts

### **Cache Structure:**
```json
{
  "cache_timestamp": "2025-07-31T11:19:54.587173",
  "datasets_cache": {
    "datasets": ["GitHub", "bi", "dataflow_errors", "serverless_pricing", "sandbox"],
    "count": 5,
    "timestamp": "2025-07-31T11:19:54.586934"
  },
  "tables_cache": {
    "bi": {
      "tables": [{"name": "table1", "type": "BASE TABLE", "created": "..."}],
      "count": 84,
      "timestamp": "..."
    }
  },
  "columns_cache": {
    "bi.my_table": {
      "columns": [{"name": "col1", "type": "STRING", "nullable": true}],
      "count": 15,
      "timestamp": "..."
    }
  }
}
```

## ⏰ Cache Expiration

### **Expiry Time:** 24 hours
- Cache automatically expires after 24 hours
- Expired cache is refreshed on next use
- You'll see: `📋 Cache expired, will refresh when needed`

### **Force Cache Refresh:**
```bash
# Method 1: Use built-in command
python3 bigquery_ai_gen_exec.py "clear-cache"

# Method 2: Delete file manually  
rm metadata_cache.json

# Method 3: Wait for auto-expiration (24 hours)
```

## 💰 Cost Impact

### **API Call Reduction:**
- **Before caching:** Every command makes API calls
- **After caching:** Only first time per 24 hours makes API calls

### **Example Savings:**
```bash
# Without caching (4 API calls)
python3 bigquery_ai_gen_exec.py "datasets"      # API call
python3 bigquery_ai_gen_exec.py "datasets"      # API call  
python3 bigquery_ai_gen_exec.py "explore bi"    # API call
python3 bigquery_ai_gen_exec.py "explore bi"    # API call

# With caching (2 API calls)
python3 bigquery_ai_gen_exec.py "datasets"      # API call (first time)
python3 bigquery_ai_gen_exec.py "datasets"      # cached (no API call)
python3 bigquery_ai_gen_exec.py "explore bi"    # API call (first time)
python3 bigquery_ai_gen_exec.py "explore bi"    # cached (no API call)
```

### **Performance Benefits:**
- **Instant responses** for cached data
- **Reduced BigQuery usage** costs
- **Better user experience** with fast exploration

## 🔍 How to Tell if Cache is Working

### **Look for these messages:**

#### **Cache Hit (No API Call):**
```
📋 Loaded cached metadata: 5 datasets, 84 tables
```

#### **Cache Miss (API Call Made):**
```
📋 Cached 5 datasets
💾 Saved metadata cache: 5 datasets, 0 tables
```

#### **File Existence:**
- Check for `metadata_cache.json` file in your directory
- File size grows as you explore more datasets/tables

## 🛠️ Troubleshooting Cache Issues

### **Check Cache Status**
```bash
python3 bigquery_ai_gen_exec.py "cache-info"
```

### **Cache Not Working:**
1. **Check file permissions:**
   ```bash
   ls -la metadata_cache.json
   # Should be readable/writable by your user
   ```

2. **Look for error messages:**
   - Watch for cache loading/saving errors in output
   - Check if disk space is available

3. **Verify BigQuery connection:**
   - Cache requires working BigQuery connection
   - Run: `gcloud auth application-default login`

### **Performance Issues:**
1. **Large cache file:**
   ```bash
   python3 bigquery_ai_gen_exec.py "cache-info"
   # Check the size - if > 10MB, consider clearing
   ```

2. **Clear cache if needed:**
   ```bash
   python3 bigquery_ai_gen_exec.py "clear-cache"
   ```

### **Manual Cache Inspection:**
```bash
# Check if file exists and size
ls -la metadata_cache.json

# Pretty print cache contents  
cat metadata_cache.json | python3 -m json.tool

# Quick size check
du -h metadata_cache.json
```

### **Common Issues & Solutions:**

| Issue | Solution |
|-------|----------|
| "Cache not available" | Check BigQuery connection |
| "Could not load cache" | Delete `metadata_cache.json` and restart |
| "Could not save cache" | Check disk space and file permissions |
| Slow performance | Use `cache-info` to check cache size |
| Stale data | Use `clear-cache` to force refresh |

## 🔧 Cache Management Commands

### **cache-info**
Shows detailed information about your current cache:
```bash
python3 bigquery_ai_gen_exec.py "cache-info"
```
**Example Output:**
```
📋 Cache Information:
   📁 File: metadata_cache.json
   💾 Size: 0.01 MB (14,063 bytes)
   🕒 Age: 2.3 hours (✅ Valid)
   📊 Cached items:
      • 5 datasets
      • 84 tables across 1 datasets
      • 0 table schemas
   ⏰ Expires after: 24 hours
   
💡 Use 'clear-cache' to clean all cached data
```

### **clear-cache**
Removes all cached metadata and forces fresh data on next use:
```bash
python3 bigquery_ai_gen_exec.py "clear-cache"
```
**Example Output:**
```
🧹 Cache cleared successfully! All metadata will be refreshed on next use.
```

**What it does:**
- ❌ Deletes `metadata_cache.json` file
- 🧹 Clears all in-memory cache
- 🔄 Next commands will make fresh API calls
- 💾 Cache will rebuild as you explore

## ✅ Summary

**Bottom Line:** The script now intelligently caches metadata to disk, so:

- **First time:** Makes API calls, caches results
- **Subsequent times:** Uses cached data (much faster, no cost)
- **Cache persists:** Across script restarts for 24 hours
- **Automatic refresh:** After 24 hours or manual deletion
