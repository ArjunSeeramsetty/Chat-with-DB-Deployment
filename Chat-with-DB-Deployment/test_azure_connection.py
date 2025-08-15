#!/usr/bin/env python3
"""
Azure SQL Connection Test Script
This script tests the connection to Azure SQL Server and verifies the configuration
"""

import os
import sys
import logging
from typing import Dict, Any
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment_file():
    """Load environment variables from .env file"""
    print("🔍 Loading environment variables...")
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    env_file = current_dir / '.env'
    
    print(f"   Looking for .env file in: {current_dir}")
    
    if not env_file.exists():
        print(f"   ❌ .env file not found at: {env_file}")
        return False
    
    print(f"   ✅ .env file found at: {env_file}")
    
    # Try to load with python-dotenv first
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print("   ✅ Environment loaded with python-dotenv")
        return True
    except ImportError:
        print("   ⚠️  python-dotenv not installed, trying manual loading...")
        
        # Manual loading as fallback
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            print("   ✅ Environment loaded manually")
            return True
        except Exception as e:
            print(f"   ❌ Failed to load .env manually: {e}")
            return False

def test_azure_connection():
    """Test Azure SQL connection using the application's database module"""
    try:
        from backend.core.database import test_database_connection, get_database_health
        from backend.core.azure_sql_utils import test_azure_connection as test_azure
        
        print("🔍 Testing Azure SQL connection...")
        print("=" * 50)
        
        # Test basic database connection
        print("1. Testing basic database connection...")
        basic_connection = test_database_connection()
        print(f"   ✅ Basic connection: {'SUCCESS' if basic_connection else 'FAILED'}")
        
        # Get detailed database health
        print("\n2. Getting database health information...")
        health = get_database_health()
        print(f"   ✅ Database type: {health.get('database_type', 'Unknown')}")
        print(f"   ✅ Connection status: {health.get('status', 'Unknown')}")
        print(f"   ✅ Is Azure SQL: {health.get('is_azure', False)}")
        
        if health.get('is_azure'):
            print(f"   ✅ Azure server: {health.get('azure_server', 'Unknown')}")
            print(f"   ✅ Azure database: {health.get('azure_database', 'Unknown')}")
            print(f"   ✅ Azure version: {health.get('azure_version', 'Unknown')}")
        
        # Test Azure-specific connection
        print("\n3. Testing Azure SQL specific features...")
        azure_test = test_azure()
        if azure_test.get('success'):
            print(f"   ✅ Azure connection: SUCCESS")
            print(f"   ✅ Server: {azure_test.get('server', 'Unknown')}")
            print(f"   ✅ Database: {azure_test.get('database', 'Unknown')}")
            print(f"   ✅ Engine Edition: {azure_test.get('engine_edition', 'Unknown')}")
            print(f"   ✅ Is Azure SQL: {azure_test.get('is_azure_sql', False)}")
        else:
            print(f"   ❌ Azure connection: FAILED")
            print(f"   ❌ Error: {azure_test.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 50)
        
        if basic_connection and azure_test.get('success'):
            print("🎉 All tests passed! Azure SQL connection is working correctly.")
            return True
        else:
            print("💥 Some tests failed. Please check your configuration.")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running this script from the project root directory.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False

def test_environment_variables():
    """Test if required environment variables are set"""
    print("🔍 Checking environment variables...")
    print("=" * 50)
    
    required_vars = [
        'DATABASE_TYPE',
        'MSSQL_SERVER',
        'MSSQL_DATABASE',
        'MSSQL_USERNAME',
        'MSSQL_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            # Mask password for security
            if var == 'MSSQL_PASSWORD':
                print(f"   ✅ {var}: {'*' * len(value)}")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            print(f"   ❌ {var}: NOT SET")
            missing_vars.append(var)
    
    print("\n" + "=" * 50)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   Please check your .env file configuration.")
        return False
    else:
        print("✅ All required environment variables are set.")
        return True

def test_odbc_driver():
    """Test if ODBC Driver 18 is available"""
    print("🔍 Checking ODBC Driver availability...")
    print("=" * 50)
    
    try:
        import pyodbc
        
        # Get available drivers
        drivers = pyodbc.drivers()
        odbc_18 = [d for d in drivers if 'ODBC Driver 18' in d]
        
        if odbc_18:
            print(f"   ✅ Found ODBC Driver 18: {odbc_18[0]}")
            return True
        else:
            print("   ❌ ODBC Driver 18 not found")
            print("   Available drivers:")
            for driver in drivers:
                print(f"     - {driver}")
            return False
            
    except ImportError:
        print("   ❌ pyodbc not installed")
        return False
    except Exception as e:
        print(f"   ❌ Error checking drivers: {e}")
        return False

def debug_environment_loading():
    """Debug environment variable loading"""
    print("🔍 Debugging environment variable loading...")
    print("=" * 50)
    
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if .env file exists
    env_file = Path('.env')
    print(f".env file exists: {env_file.exists()}")
    if env_file.exists():
        print(f".env file path: {env_file.absolute()}")
        print(f".env file size: {env_file.stat().st_size} bytes")
        
        # Show first few lines of .env file
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]
                print("First 5 lines of .env file:")
                for i, line in enumerate(lines, 1):
                    print(f"  {i}: {line.rstrip()}")
        except Exception as e:
            print(f"Error reading .env file: {e}")
    
    # Check environment variables before and after loading
    print(f"\nEnvironment variables before loading:")
    for var in ['DATABASE_TYPE', 'MSSQL_SERVER', 'MSSQL_DATABASE']:
        print(f"  {var}: {os.environ.get(var, 'NOT SET')}")
    
    # Try to load .env
    load_success = load_environment_file()
    
    print(f"\nEnvironment variables after loading:")
    for var in ['DATABASE_TYPE', 'MSSQL_SERVER', 'MSSQL_DATABASE']:
        print(f"  {var}: {os.environ.get(var, 'NOT SET')}")
    
    return load_success

def main():
    """Main test function"""
    print("🚀 Azure SQL Connection Test")
    print("=" * 60)
    
    # Debug environment loading first
    env_loaded = debug_environment_loading()
    
    if not env_loaded:
        print("\n❌ Failed to load environment variables from .env file")
        print("Please check your .env file configuration.")
        return 1
    
    # Test 1: Environment variables
    env_ok = test_environment_variables()
    
    # Test 2: ODBC Driver
    driver_ok = test_odbc_driver()
    
    # Test 3: Azure SQL connection
    connection_ok = test_azure_connection()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Environment Loading: {'✅ PASS' if env_loaded else '❌ FAIL'}")
    print(f"   Environment Variables: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"   ODBC Driver: {'✅ PASS' if driver_ok else '❌ FAIL'}")
    print(f"   Azure SQL Connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    
    if all([env_loaded, env_ok, driver_ok, connection_ok]):
        print("\n🎉 All tests passed! Your Azure SQL configuration is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
