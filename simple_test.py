#!/usr/bin/env python3
"""
Simple test to debug SQL execution
"""

import sqlite3

def test_simple_sql():
    """Test simple SQL execution"""
    
    print("Testing simple SQL execution...")
    
    try:
        # Connect to database
        conn = sqlite3.connect('C:/Users/arjun/Desktop/PSPreport/power_data.db')
        cursor = conn.cursor()
        
        # Test simple query first
        cursor.execute("SELECT COUNT(*) FROM FactAllIndiaDailySummary WHERE Year = 2024")
        count = cursor.fetchone()[0]
        print(f"Simple query result: {count} rows in 2024")
        
        # Test the growth SQL
        sql = """WITH MonthlyData AS (
            SELECT dt.Year, dt.Month, RegionName, ROUND(SUM(f.EnergyShortage), 2) as MonthlyValue
            FROM FactAllIndiaDailySummary f
            JOIN DimDates dt ON f.DateID = dt.DateID
            JOIN DimRegions d ON f.RegionID = d.RegionID
            WHERE dt.Year = 2024
            GROUP BY dt.Year, dt.Month, RegionName
            ORDER BY dt.Year, dt.Month, RegionName
        )
        SELECT COUNT(*) FROM MonthlyData"""
        
        cursor.execute(sql)
        count = cursor.fetchone()[0]
        print(f"Growth SQL result: {count} rows in MonthlyData")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_sql()
    print(f"Test {'PASSED' if success else 'FAILED'}") 