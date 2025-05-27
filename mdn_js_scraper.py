#!/usr/bin/env python3
"""
MDN JavaScript Documentation Scraper
-----------------------------------
This script scrapes the entire JavaScript documentation from Mozilla Developer Network (MDN)
and organizes it into a comprehensive JSON tree structure.

Created: 2025-05-26
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import random
from urllib.parse import urljoin, urlparse
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mdn_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MDNJSScraper:
    """A scraper to extract all JavaScript documentation from MDN."""
    
    BASE_URL = "https://developer.mozilla.org"
    JS_DOCS_URL = f"{BASE_URL}/en-US/docs/Web/JavaScript"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://developer.mozilla.org/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    def __init__(self, output_file="mdn_javascript_docs.json", max_threads=5, delay_range=(1, 3)):
        """
        Initialize the MDN JavaScript scraper.
        
        Args:
            output_file (str): Path to the output JSON file
            max_threads (int): Maximum number of concurrent threads for scraping
            delay_range (tuple): Range for random delay between requests in seconds
        """
        self.output_file = output_file
        self.max_threads = max_threads
        self.delay_range = delay_range
        self.visited_urls = set()
        self.js_tree = defaultdict(dict)
        self.session = requests.Session()
        
        # Create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)
        
    def request_page(self, url):
        """
        Make an HTTP request to the given URL with proper handling.
        
        Args:
            url (str): The URL to request
            
        Returns:
            BeautifulSoup: Parsed HTML content or None if failed
        """
        try:
            # Add random delay to avoid overloading the server
            time.sleep(random.uniform(*self.delay_range))
            
            response = self.session.get(url, headers=self.HEADERS, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                return soup
            else:
                logger.warning(f"Failed to fetch {url}, status code: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
    
    def get_page_content(self, url):
        """
        Extract structured content from an MDN page.
        
        Args:
            url (str): URL of the page to scrape
            
        Returns:
            dict: Structured content of the page
        """
        # Skip if already visited
        if url in self.visited_urls:
            return {}
        
        self.visited_urls.add(url)
        logger.info(f"Processing: {url}")
        
        soup = self.request_page(url)
        if not soup:
            return {}
        
        # Extract page metadata
        content = {
            "url": url,
            "title": self._extract_title(soup),
            "description": self._extract_description(soup),
            "syntax": self._extract_syntax(soup),
            "parameters": self._extract_parameters(soup),
            "return_value": self._extract_return_value(soup),
            "examples": self._extract_examples(soup),
            "browser_compatibility": self._extract_browser_compatibility(soup),
            "see_also": self._extract_see_also(soup),
            # Add the new extraction:
            **self._extract_inheritance_and_properties(soup),
            "children": []
        }
        
        # Find child pages and add to structure
        child_links = self._find_child_links(soup, url)
        logger.info(f"Found {len(child_links)} child links on {url}")  # Add this line
        
        return {
            "content": content,
            "child_links": child_links
        }
    
    def _extract_title(self, soup):
        """Extract the title of the page."""
        try:
            return soup.find('h1').text.strip()
        except (AttributeError, TypeError):
            return ""
    
    def _extract_description(self, soup):
        """Extract the main description from the page."""
        try:
            article = soup.find('article')
            if article:
                # Get the first paragraph after the title
                description_paras = []
                for p in article.find_all('p'):
                    if p.text.strip():  # Non-empty paragraph
                        description_paras.append(p.text.strip())
                        
                return "\n\n".join(description_paras[:3])  # First 3 paragraphs
            return ""
        except (AttributeError, TypeError):
            return ""
    
    def _extract_syntax(self, soup):
        """Extract syntax information."""
        syntax = []
        try:
            syntax_section = soup.find('h2', string=lambda text: text and 'Syntax' in text)
            if syntax_section:
                # Look for code blocks after the Syntax heading
                next_elem = syntax_section.find_next_sibling()
                while next_elem and next_elem.name != 'h2':
                    if next_elem.name == 'pre':
                        syntax.append(next_elem.text.strip())
                    next_elem = next_elem.find_next_sibling()
        except Exception as e:
            logger.debug(f"Error extracting syntax: {str(e)}")
        
        return syntax
    
    def _extract_parameters(self, soup):
        """Extract parameters information."""
        parameters = []
        try:
            params_section = soup.find(['h2', 'h3'], string=lambda text: text and ('Parameters' in text or 'Arguments' in text))
            if params_section:
                # Find parameter list or description
                param_list = params_section.find_next('dl')
                if param_list:
                    for dt, dd in zip(param_list.find_all('dt'), param_list.find_all('dd')):
                        parameters.append({
                            "name": dt.text.strip(),
                            "description": dd.text.strip()
                        })
                else:
                    # Try finding a paragraph or list describing parameters
                    next_elem = params_section.find_next_sibling()
                    while next_elem and next_elem.name != 'h2' and next_elem.name != 'h3':
                        if next_elem.name in ['p', 'ul', 'ol']:
                            parameters.append({
                                "name": "General",
                                "description": next_elem.text.strip()
                            })
                        next_elem = next_elem.find_next_sibling()
        except Exception as e:
            logger.debug(f"Error extracting parameters: {str(e)}")
        
        return parameters
    
    def _extract_return_value(self, soup):
        """Extract return value information."""
        try:
            return_section = soup.find(['h2', 'h3'], string=lambda text: text and 'Return' in text and 'value' in text.lower())
            if return_section:
                return_info = ""
                next_elem = return_section.find_next_sibling()
                while next_elem and next_elem.name != 'h2' and next_elem.name != 'h3':
                    if next_elem.name in ['p', 'ul', 'ol']:
                        return_info += next_elem.text.strip() + "\n\n"
                    next_elem = next_elem.find_next_sibling()
                return return_info.strip()
        except Exception as e:
            logger.debug(f"Error extracting return value: {str(e)}")
        
        return ""
    
    def _extract_examples(self, soup):
        """Extract code examples."""
        examples = []
        try:
            examples_section = soup.find(['h2', 'h3'], string=lambda text: text and 'Example' in text)
            if examples_section:
                next_elem = examples_section.find_next_sibling()
                current_example = {"title": "", "description": "", "code": ""}
                
                while next_elem and next_elem.name != 'h2':
                    if next_elem.name == 'h3':
                        # Save previous example if it has content
                        if current_example["code"] or current_example["description"]:
                            examples.append(current_example)
                        # Start a new example
                        current_example = {"title": next_elem.text.strip(), "description": "", "code": ""}
                    
                    elif next_elem.name == 'pre':
                        current_example["code"] += next_elem.text.strip() + "\n"
                    
                    elif next_elem.name in ['p', 'ul', 'ol']:
                        current_example["description"] += next_elem.text.strip() + "\n"
                    
                    next_elem = next_elem.find_next_sibling()
                
                # Add the last example if it has content
                if current_example["code"] or current_example["description"]:
                    examples.append(current_example)
        except Exception as e:
            logger.debug(f"Error extracting examples: {str(e)}")
        
        return examples
    
    def _extract_browser_compatibility(self, soup):
        """Extract browser compatibility information."""
        compatibility = {}
        try:
            compat_section = soup.find('h2', id=lambda id_: id_ and 'browser_compatibility' in id_)
            if compat_section:
                compat_table = compat_section.find_next('table')
                if compat_table:
                    # Process the compatibility table
                    browsers = []
                    for th in compat_table.find_all('th'):
                        if th.text.strip() and th.text.strip() != "Feature":
                            browsers.append(th.text.strip())
                    
                    for row in compat_table.find_all('tr')[1:]:  # Skip header row
                        cells = row.find_all(['th', 'td'])
                        if cells:
                            feature = cells[0].text.strip()
                            feature_compat = {}
                            
                            for i, browser in enumerate(browsers):
                                if i + 1 < len(cells):
                                    feature_compat[browser] = cells[i + 1].text.strip()
                            
                            compatibility[feature] = feature_compat
        except Exception as e:
            logger.debug(f"Error extracting compatibility: {str(e)}")
        
        return compatibility
    
    def _extract_see_also(self, soup):
        """Extract 'See also' links."""
        see_also = []
        try:
            see_also_section = soup.find('h2', string=lambda text: text and 'See also' in text)
            if see_also_section:
                links_section = see_also_section.find_next_sibling()
                if links_section and links_section.name == 'ul':
                    for li in links_section.find_all('li'):
                        link = li.find('a')
                        if link and link.get('href'):
                            href = link.get('href')
                            # Make sure URL is absolute
                            if not href.startswith(('http://', 'https://')):
                                href = urljoin(self.BASE_URL, href)
                            
                            see_also.append({
                                "text": link.text.strip(),
                                "url": href
                            })
        except Exception as e:
            logger.debug(f"Error extracting see also links: {str(e)}")
        
        return see_also
    
    def _extract_inheritance_and_properties(self, soup):
        """Extract inheritance chain and object properties if available."""
        result = {
            "inheritance": [],
            "properties": [],
            "methods": []
        }
        
        try:
            # Look for inheritance information
            inheritance_section = soup.find(['h2', 'h3'], string=lambda text: text and ('Inheritance' in text))
            if inheritance_section:
                # Often shown as a list or diagram
                next_elem = inheritance_section.find_next_sibling()
                while next_elem and next_elem.name not in ['h2', 'h3']:
                    if next_elem.name in ['ul', 'ol']:
                        for item in next_elem.find_all('li'):
                            result["inheritance"].append(item.text.strip())
                    next_elem = next_elem.find_next_sibling()
            
            # Look for properties table/list
            properties_section = soup.find(['h2', 'h3'], string=lambda text: text and ('Properties' in text))
            if properties_section:
                next_elem = properties_section.find_next_sibling()
                while next_elem and next_elem.name not in ['h2', 'h3']:
                    if next_elem.name in ['dl', 'ul', 'table']:
                        # For definition lists
                        if next_elem.name == 'dl':
                            for dt in next_elem.find_all('dt'):
                                result["properties"].append({
                                    "name": dt.text.strip(),
                                    "description": dt.find_next('dd').text.strip() if dt.find_next('dd') else ""
                                })
                        # For tables
                        elif next_elem.name == 'table':
                            for row in next_elem.find_all('tr')[1:]:  # Skip header
                                cells = row.find_all(['td', 'th'])
                                if len(cells) >= 2:
                                    result["properties"].append({
                                        "name": cells[0].text.strip(),
                                        "description": cells[1].text.strip()
                                    })
                    next_elem = next_elem.find_next_sibling()
                    
            # Look for methods table/list
            methods_section = soup.find(['h2', 'h3'], string=lambda text: text and ('Methods' in text))
            if methods_section:
                # Similar to properties extraction
                next_elem = methods_section.find_next_sibling()
                while next_elem and next_elem.name not in ['h2', 'h3']:
                    if next_elem.name in ['dl', 'ul', 'table']:
                        # Extract method information similar to properties
                        if next_elem.name == 'dl':
                            for dt in next_elem.find_all('dt'):
                                result["methods"].append({
                                    "name": dt.text.strip(),
                                    "description": dt.find_next('dd').text.strip() if dt.find_next('dd') else ""
                                })
                    next_elem = next_elem.find_next_sibling()
                    
        except Exception as e:
            logger.debug(f"Error extracting inheritance/properties: {str(e)}")
        
        return result
    
    def _find_child_links(self, soup, current_url):
        """Find links to child pages anywhere in the article."""
        child_links = set()
        try:
            # More aggressive link finding - search ALL links in the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Broaden link detection to catch more JS doc pages
                if '/docs/Web/JavaScript/' in href or 'JavaScript/Reference' in href:
                    # Ensure we get the full URL
                    abs_url = urljoin(self.BASE_URL, href)
                    if abs_url not in self.visited_urls and self._is_js_doc_page(abs_url):
                        child_links.add(abs_url)
                        # Debug - print each link found
                        logger.debug(f"Found child link: {abs_url}")
                    
        except Exception as e:
            logger.error(f"Error finding child links: {str(e)}")
    
        return list(child_links)
    
    def _is_js_doc_page(self, url):
        """Check if the URL is a JavaScript documentation page."""
        parsed = urlparse(url)
        path = parsed.path
        
        # Check if this is a JavaScript documentation page
        if '/docs/Web/JavaScript' in path:
            # Exclude versioned docs
            if '/docs/Web/JavaScript/New_in_JavaScript' in path:
                return False
                
            # Exclude non-documentation pages
            excluded_patterns = [
                'index.html',
                'contributors.txt',
                '/tag/',
                '/docs/MDN/',
            ]
            
            for pattern in excluded_patterns:
                if pattern in path:
                    return False
                    
            return True
            
        return False
    
    def build_tree(self, url=None):
        """
        Build the JavaScript documentation tree starting from the specified URL.
        
        Args:
            url (str, optional): Starting URL. If None, use the default JS docs URL.
            
        Returns:
            dict: The complete JavaScript documentation tree
        """
        if url is None:
            url = self.JS_DOCS_URL

        queue = [url]
        processed_urls = set()

        # Special handling for root - manually add key JS sections
        if url == self.JS_DOCS_URL:
            # Add important JavaScript sections manually
            key_sections = [
                "/en-US/docs/Web/JavaScript/Reference",
                "/en-US/docs/Web/JavaScript/Reference/Global_Objects",
                "/en-US/docs/Web/JavaScript/Guide",
                "/en-US/docs/Web/JavaScript/Reference/Operators",
                "/en-US/docs/Web/JavaScript/Reference/Statements",
                "/en-US/docs/Web/JavaScript/Reference/Functions",
                "/en-US/docs/Web/JavaScript/Reference/Classes"
            ]
            for section in key_sections:
                abs_url = urljoin(self.BASE_URL, section)
                if abs_url not in queue:
                    queue.append(abs_url)
                    logger.info(f"Added key section: {abs_url}")
                
            # Also find links on the root page
            soup = self.request_page(url)
            if soup:
                # Add the root page content to the tree
                root_data = self.get_page_content(url)
                if root_data and 'content' in root_data:
                    self._add_to_tree(url, root_data['content'])
                
                # Find all links to main JS sections and subpages
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if href.startswith('/en-US/docs/Web/JavaScript/') and href != '/en-US/docs/Web/JavaScript':
                        abs_url = urljoin(self.BASE_URL, href)
                        if abs_url not in queue:
                            queue.append(abs_url)
        
        # Process queue for breadth-first traversal
        while queue:
            current_batch = []
            
            # Prepare current batch of URLs to process
            while queue and len(current_batch) < self.max_threads:
                current_url = queue.pop(0)
                if current_url not in processed_urls:
                    current_batch.append(current_url)
                    processed_urls.add(current_url)
            
            if not current_batch:
                continue
                
            # Process current batch with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_url = {
                    executor.submit(self.get_page_content, url): url 
                    for url in current_batch
                }
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        data = future.result()
                        if data:
                            # Add to tree
                            self._add_to_tree(url, data['content'])
                            
                            # Add child links to queue
                            for child_url in data.get('child_links', []):
                                if child_url not in processed_urls and child_url not in queue:
                                    queue.append(child_url)
                    except Exception as e:
                        logger.error(f"Error processing {url}: {str(e)}")
            
            # Save checkpoint after each batch
            self._save_checkpoint()
            
        return self.js_tree
    
    def _add_to_tree(self, url, content):
        """
        Add page content to the appropriate place in the tree structure.
        
        Args:
            url (str): URL of the page
            content (dict): Structured content from the page
        """
        # Parse the URL to determine the category hierarchy
        path = urlparse(url).path
        parts = [p for p in path.split('/') if p and p not in ['en-US', 'docs', 'Web']]
        
        # Skip non-JavaScript pages
        if not parts or parts[0] != 'JavaScript':
            return
            
        # Remove 'JavaScript' from the beginning
        parts = parts[1:]
        
        # Add content to the tree at the appropriate level
        current = self.js_tree
        for i, part in enumerate(parts):
            # Clean up part name for use as a key
            clean_part = self._clean_key_name(part)
            
            # If this is the last part, add the content
            if i == len(parts) - 1:
                current[clean_part] = content
            else:
                # Create path if it doesn't exist
                if clean_part not in current:
                    current[clean_part] = {}
                current = current[clean_part]
    
    def _clean_key_name(self, name):
        """
        Clean up a name for use as a key in the tree.
        
        Args:
            name (str): Original name
            
        Returns:
            str: Cleaned name
        """
        # Remove file extension if present
        name = re.sub(r'\.html$', '', name)
        
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        
        return name
    
    def _save_checkpoint(self):
        """Save a checkpoint of the current progress."""
        checkpoint_path = os.path.join("checkpoints", f"checkpoint_{int(time.time())}.json")
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(self.js_tree, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def save_tree(self):
        """Save the complete tree to the output file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.js_tree, f, indent=2)
        
        logger.info(f"Saved complete JavaScript documentation tree to: {self.output_file}")
        
        # Generate a statistics summary
        stats = self._generate_stats()
        stats_file = os.path.splitext(self.output_file)[0] + "_stats.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Saved statistics to: {stats_file}")
        
        return self.output_file
    
    def _generate_stats(self):
        """Generate statistics about the scraped data."""
        stats = {
            "total_pages": len(self.visited_urls),
            "scrape_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories": {},
            "structure_depth": self._get_max_depth(self.js_tree),
        }
        
        # Count items in each top-level category
        for category, content in self.js_tree.items():
            stats["categories"][category] = self._count_items(content)
        
        return stats
    
    def _count_items(self, obj):
        """Count the number of items in a nested dictionary."""
        if isinstance(obj, dict):
            # If it's a page content dict with specific keys
            if "url" in obj and "title" in obj:
                return 1
            
            # Otherwise it's a category containing other items
            return sum(self._count_items(val) for val in obj.values())
        
        return 0
    
    def _get_max_depth(self, obj, current_depth=0):
        """Get the maximum depth of the tree."""
        if not isinstance(obj, dict):
            return current_depth
            
        if not obj:
            return current_depth
            
        # If this looks like a content object rather than a container
        if "url" in obj and "title" in obj:
            return current_depth
            
        # Otherwise, recurse into children
        return max(
            self._get_max_depth(child, current_depth + 1)
            for child in obj.values()
        )


if __name__ == "__main__":
    # Create and run the scraper
    scraper = MDNJSScraper(
        output_file="mdn_javascript_complete.json",
        max_threads=5,  # Adjust based on your system capabilities
        delay_range=(1, 3)  # Random delay between requests to avoid rate limiting
    )
    
    print("Starting MDN JavaScript documentation scraping...")
    print("This may take several hours to complete due to the comprehensive nature and politeness delays.")
    print("Progress will be logged to mdn_scraper.log")
    print("Checkpoints will be saved periodically to the 'checkpoints' directory.")
    
    start_time = time.time()
    
    try:
        # Build the documentation tree
        tree = scraper.build_tree()
        
        # Save the results
        output_file = scraper.save_tree()
        
        elapsed_time = (time.time() - start_time) / 60  # in minutes
        
        print(f"\nScraping completed successfully in {elapsed_time:.2f} minutes!")
        print(f"JavaScript documentation tree saved to: {output_file}")
        print(f"Statistics saved to: {os.path.splitext(output_file)[0]}_stats.json")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        print("Saving the data collected so far...")
            
        # Save whatever we have
        scraper.save_tree()
        
        print("Partial data has been saved. You can resume by running the script again.")
        
    except Exception as e:
        print(f"\nError during scraping: {str(e)}")
        print("Saving the data collected so far...")
        
        # Try to save what we have
        try:
            scraper.save_tree()
            print("Partial data has been saved.")
        except Exception as e:
            print(f"Failed to save partial data: {e}")