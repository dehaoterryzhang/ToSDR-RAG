#!/usr/bin/env python3
"""
Simple dashboard to view and analyze user feedback data.
"""

import json
import os
from datetime import datetime
from collections import Counter

def load_feedback_data():
    """Load feedback data from the log file."""
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    feedback_log_file = os.path.join(log_dir, "feedback.jsonl")
    
    feedback_data = []
    
    if os.path.exists(feedback_log_file):
        with open(feedback_log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        feedback_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line: {line}")
    else:
        print(f"No feedback log file found at: {feedback_log_file}")
    
    return feedback_data

def display_feedback_summary(feedback_data):
    """Display a summary of the feedback data."""
    if not feedback_data:
        print("No feedback data available.")
        return
    
    print("="*60)
    print("FEEDBACK DASHBOARD SUMMARY")
    print("="*60)
    
    # Total feedback count
    total_feedback = len(feedback_data)
    print(f"Total feedback entries: {total_feedback}")
    
    # Rating distribution
    ratings = [entry['rating'] for entry in feedback_data]
    rating_counts = Counter(ratings)
    
    thumbs_up = rating_counts.get('thumbs_up', 0)
    thumbs_down = rating_counts.get('thumbs_down', 0)
    
    print(f"👍 Thumbs up: {thumbs_up} ({thumbs_up/total_feedback*100:.1f}%)")
    print(f"👎 Thumbs down: {thumbs_down} ({thumbs_down/total_feedback*100:.1f}%)")
    
    # Date range
    if feedback_data:
        dates = [entry['timestamp'] for entry in feedback_data]
        dates.sort()
        print(f"Feedback date range: {dates[0][:10]} to {dates[-1][:10]}")
    
    print("\n" + "="*60)
    print("RECENT FEEDBACK (Last 10 entries)")
    print("="*60)
    
    # Show recent feedback
    recent_feedback = feedback_data[-10:]  # Last 10 entries
    
    for i, entry in enumerate(reversed(recent_feedback), 1):
        timestamp = entry['timestamp'][:19]  # Remove microseconds
        rating_emoji = "👍" if entry['rating'] == 'thumbs_up' else "👎"
        query_preview = entry['query'][:50] + "..." if len(entry['query']) > 50 else entry['query']
        
        print(f"\n{i}. {timestamp} - {rating_emoji}")
        print(f"   Query: {query_preview}")
        print(f"   Answer preview: {entry['answer'][:100]}...")

def main():
    """Main function to run the dashboard."""
    feedback_data = load_feedback_data()
    display_feedback_summary(feedback_data)
    
    if feedback_data:
        print("\n" + "="*60)
        print("EXPORT OPTIONS")
        print("="*60)
        print("To export all feedback data to CSV:")
        print("python feedback_dashboard.py --export-csv")
        print("\nTo view detailed feedback:")
        print("python feedback_dashboard.py --detailed")

def export_to_csv(feedback_data):
    """Export feedback data to CSV format."""
    import csv
    
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    csv_file = os.path.join(log_dir, "feedback_export.csv")
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        if feedback_data:
            fieldnames = feedback_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(feedback_data)
    
    print(f"Feedback data exported to: {csv_file}")

def show_detailed_feedback(feedback_data):
    """Show detailed view of all feedback."""
    print("="*60)
    print("DETAILED FEEDBACK VIEW")
    print("="*60)
    
    for i, entry in enumerate(feedback_data, 1):
        print(f"\nEntry #{i}")
        print(f"Timestamp: {entry['timestamp']}")
        print(f"Rating: {'👍 Thumbs up' if entry['rating'] == 'thumbs_up' else '👎 Thumbs down'}")
        print(f"Query: {entry['query']}")
        print(f"Answer: {entry['answer']}")
        print("-" * 40)

if __name__ == "__main__":
    import sys
    
    feedback_data = load_feedback_data()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--export-csv":
            export_to_csv(feedback_data)
        elif sys.argv[1] == "--detailed":
            show_detailed_feedback(feedback_data)
        else:
            print("Unknown option. Use --export-csv or --detailed")
    else:
        main()