#!/usr/bin/env python3
"""Script to inspect the Chunkenizer SQLite database."""
import sqlite3
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings


def format_datetime(dt_str):
    """Format datetime string for display."""
    if dt_str:
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return dt_str
    return 'N/A'


def print_table(conn, table_name):
    """Print all rows from a table."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    
    if not rows:
        print(f"\n{table_name}: No rows found")
        return
    
    print(f"\n{'='*80}")
    print(f"Table: {table_name} ({len(rows)} rows)")
    print('='*80)
    
    # Print column headers
    header = " | ".join(f"{col:20}" for col in columns)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for row in rows:
        formatted_row = []
        for i, val in enumerate(row):
            if columns[i] in ['created_at', 'updated_at']:
                formatted_row.append(f"{format_datetime(val):20}")
            elif columns[i] == 'metadata_json' and val:
                try:
                    metadata = json.loads(val)
                    formatted_row.append(f"{json.dumps(metadata, indent=0)[:50]:20}")
                except:
                    formatted_row.append(f"{str(val)[:50]:20}")
            else:
                formatted_row.append(f"{str(val)[:50]:20}")
        print(" | ".join(formatted_row))


def show_statistics(conn):
    """Show database statistics."""
    cursor = conn.cursor()
    
    # Count documents
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    
    # Total chunks
    cursor.execute("SELECT SUM(chunk_count) FROM documents")
    total_chunks = cursor.fetchone()[0] or 0
    
    # Total tokens
    cursor.execute("SELECT SUM(total_tokens) FROM documents")
    total_tokens = cursor.fetchone()[0] or 0
    
    # Average chunks per document
    cursor.execute("SELECT AVG(chunk_count) FROM documents")
    avg_chunks = cursor.fetchone()[0] or 0
    
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
    print(f"Total Documents:     {doc_count}")
    print(f"Total Chunks:        {total_chunks}")
    print(f"Total Tokens:        {total_tokens:,}")
    print(f"Avg Chunks/Doc:      {avg_chunks:.2f}")
    print("="*80)


def show_schema(conn):
    """Show database schema."""
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("\n" + "="*80)
    print("DATABASE SCHEMA")
    print("="*80)
    for table in tables:
        print(table[0])
        print()


def main():
    """Main function."""
    db_path = Path(settings.sqlite_path).expanduser().resolve()
    
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        sys.exit(1)
    
    print(f"Connecting to database: {db_path}")
    print(f"File size: {db_path.stat().st_size / 1024:.2f} KB")
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    try:
        # Show schema
        show_schema(conn)
        
        # Show statistics
        show_statistics(conn)
        
        # Show documents table
        print_table(conn, "documents")
        
        # Interactive mode
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("Enter SQL queries (or 'quit' to exit):")
        print("Examples:")
        print("  SELECT name, chunk_count, total_tokens FROM documents LIMIT 5;")
        print("  SELECT * FROM documents WHERE name LIKE '%test%';")
        print()
        
        cursor = conn.cursor()
        while True:
            try:
                query = input("SQL> ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                cursor.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    rows = cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    if rows:
                        # Print headers
                        print(" | ".join(f"{col:20}" for col in columns))
                        print("-" * (len(columns) * 23))
                        
                        # Print rows
                        for row in rows:
                            print(" | ".join(f"{str(val)[:20]:20}" for val in row))
                        print(f"\n{len(rows)} row(s)")
                    else:
                        print("No results")
                else:
                    conn.commit()
                    print("Query executed successfully")
                    
            except sqlite3.Error as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()

