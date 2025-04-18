"""
Bloomberg News API plugin that fetches and formats data into RSSGuard-compatible JSON

Example usage:
    plugin = Bloomberg()
    result = plugin.execute_pipeline_step({
        "config": {
            "api_url": "https://feeds.bloomberg.com/news.json",
            "params": {
                "ageHours": 120,
                "token": "your_token",
                "tickers": "NTRS:US"
            }
        },
        "output": "bloomberg_feed"
    }, {})
"""

import requests
import json
import datetime
from Plugins.BasePlugin import BasePlugin


class Bloomberg(BasePlugin):
    """Bloomberg News API plugin that fetches and formats data into RSSGuard-compatible JSON"""

    plugin_type = "Input"

    def __init__(self):
        """Initialize the Bloomberg plugin"""
        self.base_url = "https://feeds.bloomberg.com/news.json"

    def execute_pipeline_step(self, step_config, context):
        """Execute a pipeline step for this plugin
        
        Expected step_config format:
        {
            "plugin": "Bloomberg",
            "config": {
                "api_url": "https://feeds.bloomberg.com/news.json",  # Optional
                "params": {
                    "ageHours": 120,
                    "token": "your_token",
                    "tickers": "NTRS:US"
                }
            },
            "output": "bloomberg_feed"
        }
        """
        config = step_config["config"]
        
        # Get API URL and params
        api_url = config.get("api_url", self.base_url)
        params = config.get("params", {})
        
        # Fetch and format data
        feed_data = self.get_feed(api_url, params)
        
        return {step_config["output"]: feed_data}

    def get_feed(self, api_url, params=None):
        """
        Fetches data from the Bloomberg API and converts it to RSSGuard-compatible JSON
        
        Args:
            api_url (str): The Bloomberg API URL to fetch data from
            params (dict): Optional query parameters
            
        Returns:
            dict: RSSGuard-compatible JSON feed data
            
        Raises:
            ValueError: If there's an error fetching or parsing the data
        """
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error fetching data from Bloomberg API: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing Bloomberg API response: {str(e)}")

        # Format response as RSSGuard feed
        rssguard_data = {
            "version": 1,
            "title": "Bloomberg News Feed",
            "link": "https://www.bloomberg.com/",
            "description": "Bloomberg News",
            "items": []
        }

        for item in data.get("items", []):
            rss_item = {
                "title": item.get("title", "No Title"),
                "link": item.get("link", ""),
                "guid": item.get("id", ""),
                "description": item.get("description", ""),
                "pubDate": item.get("pubDate", datetime.datetime.now().isoformat()),
                "author": item.get("author", "Bloomberg"),
                "categories": item.get("categories", []),
                "tickers": item.get("tickers", [])
            }
            rssguard_data["items"].append(rss_item)

        return rssguard_data


if __name__ == "__main__":
    # Test the plugin
    plugin = Bloomberg()
    
    # Test configuration
    test_config = {
        "plugin": "Bloomberg",
        "config": {
            "params": {
                "ageHours": 120,
                "token": "glassdoor:gd4bloomberg",
                "tickers": "NTRS:US"
            }
        },
        "output": "feed"
    }
    
    try:
        result = plugin.execute_pipeline_step(test_config, {})
        feed = result["feed"]
        
        print(f"Feed Title: {feed['title']}")
        print(f"Number of items: {len(feed['items'])}")
        
        # Print first 3 items
        for item in feed["items"][:3]:
            print(f"\nTitle: {item['title']}")
            print(f"Link: {item['link']}")
            print(f"Author: {item['author']}")
            if item.get("tickers"):
                print(f"Tickers: {', '.join(item['tickers'])}")
                
    except ValueError as e:
        print(f"Error: {e}")