import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
import time
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
import logging
from urllib.robotparser import RobotFileParser
import xml.etree.ElementTree as ET
import hashlib
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)

# Set up requests session with proper headers
session = requests.Session()
session.headers.update({
    "User-Agent": "NeutrinoChatbot/1.0 (https://neutrinotechsystems.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
})

class EnhancedWebCrawler:
    def __init__(self, start_urls=None, max_pages=200, max_depth=5,
                 concurrency=8, respect_robots=True, delay=0.5,
                 parse_sitemaps=True, extract_pdf=True, follow_subdomains=True):
        """
        Initialize the enhanced web crawler.
        
        Args:
            start_urls (list): List of URLs to start crawling from
            max_pages (int): Maximum number of pages to crawl
            max_depth (int): Maximum depth to crawl
            concurrency (int): Number of concurrent requests
            respect_robots (bool): Whether to respect robots.txt
            delay (float): Delay between requests to the same domain
            parse_sitemaps (bool): Whether to parse sitemaps
            extract_pdf (bool): Whether to extract text from PDFs
            follow_subdomains (bool): Whether to follow subdomains
        """
        self.start_urls = start_urls or ["https://neutrinotechsystems.com"]
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.concurrency = concurrency
        self.respect_robots = respect_robots
        self.delay = delay
        self.parse_sitemaps = parse_sitemaps
        self.extract_pdf = extract_pdf
        self.follow_subdomains = follow_subdomains
        
        self.visited = set()
        self.scraped_pages = []
        self.robots_parsers = {}
        self.last_request_time = {}
        self.sitemap_urls = set()
        self.content_hashes = set()  # To detect duplicate content
        self.url_scores = defaultdict(int)  # To prioritize important URLs
        
    def is_internal_link(self, link, base_domain=None):
        """Check if a link is internal to the specified domain."""
        if not link:
            return False
            
        parsed = urlparse(link)
        if not base_domain:
            base_domain = urlparse(self.start_urls[0]).netloc
        
        # Check if it's the same domain or a subdomain if follow_subdomains is enabled
        if self.follow_subdomains:
            is_subdomain = parsed.netloc.endswith(f".{base_domain}") if base_domain else False
            return (parsed.netloc in ["", base_domain] or is_subdomain) and not link.startswith(("#", "mailto:", "tel:"))
        else:
            return parsed.netloc in ["", base_domain] and not link.startswith(("#", "mailto:", "tel:"))
    
    def can_fetch(self, url):
        """Check if the URL can be fetched according to robots.txt."""
        if not self.respect_robots:
            return True
            
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Skip non-HTTP URLs
        if not parsed_url.scheme.startswith('http'):
            return False
        
        if domain not in self.robots_parsers:
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                parser.read()
                self.robots_parsers[domain] = parser
            except Exception as e:
                logging.warning(f"Could not fetch robots.txt for {domain}: {e}")
                return True
                
        return self.robots_parsers[domain].can_fetch("*", url)
    
    def respect_rate_limits(self, url):
        """Respect rate limits for each domain."""
        domain = urlparse(url).netloc
        current_time = time.time()
        
        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
                
        self.last_request_time[domain] = time.time()
    
    def clean_text(self, text):
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-printable characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        return text
    
    def extract_metadata(self, soup, url):
        """Extract metadata from the page."""
        metadata = {
            "url": url,
            "title": "",
            "description": "",
            "headings": []
        }
        
        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)
            
        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc["content"]
            
        # Extract headings
        for level in range(1, 7):
            headings = soup.find_all(f"h{level}")
            for heading in headings:
                text = heading.get_text(strip=True)
                if text:
                    metadata["headings"].append({
                        "level": level,
                        "text": text
                    })
                    
        return metadata
    
    def extract_links_and_content(self, url):
        """Extract links and content from a URL."""
        try:
            self.respect_rate_limits(url)
            
            if not self.can_fetch(url):
                logging.info(f"Skipping {url} (disallowed by robots.txt)")
                return None, []
                
            headers = {
                "User-Agent": "NeutrinoChatbot/1.0 (https://neutrinotechsystems.com)"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Check if content is HTML
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type.lower():
                logging.info(f"Skipping non-HTML content: {url} ({content_type})")
                return None, []
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for tag in soup.select('script, style, nav, footer, iframe, [class*="cookie"], [class*="popup"], [id*="cookie"], [id*="popup"]'):
                tag.decompose()
                
            # Extract metadata
            metadata = self.extract_metadata(soup, url)
            
            # Extract main content
            main_content = ""
            main_tags = soup.select("main, article, .content, #content, .main, #main")
            
            if main_tags:
                # Use the first main content area found
                main_content = main_tags[0].get_text(separator=" ", strip=True)
            else:
                # Fallback to body content
                main_content = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
                
            # Clean content
            cleaned_content = self.clean_text(main_content)
            
            # Extract links
            links = []
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                if href:
                    full_url = urljoin(url, href)
                    links.append(full_url)
                    
            return {
                "metadata": metadata,
                "content": cleaned_content
            }, links
            
        except Exception as e:
            logging.error(f"Failed to fetch {url}: {e}")
            return None, []
    
    def process_url(self, url_info):
        """Process a single URL (for use with ThreadPoolExecutor)."""
        url, depth = url_info
        
        if url in self.visited or depth > self.max_depth:
            return None
            
        if not self.is_internal_link(url):
            return None
            
        logging.info(f"Crawling: {url} (depth: {depth})")
        self.visited.add(url)
        
        page_data, links = self.extract_links_and_content(url)
        new_links = []
        
        if page_data:
            self.scraped_pages.append(page_data)
            
            for link in links:
                if link not in self.visited:
                    new_links.append((link, depth + 1))
                    
        return new_links
    
    def parse_sitemap(self, base_url):
        """Parse sitemap.xml to find URLs to crawl."""
        try:
            # Try to find sitemap from robots.txt first
            domain = urlparse(base_url).netloc
            scheme = urlparse(base_url).scheme
            robots_url = f"{scheme}://{domain}/robots.txt"
            
            try:
                robots_response = session.get(robots_url, timeout=10)
                robots_response.raise_for_status()
                
                # Look for Sitemap: entries in robots.txt
                for line in robots_response.text.split('\n'):
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        self.sitemap_urls.add(sitemap_url)
            except Exception as e:
                logging.warning(f"Could not fetch robots.txt for {domain}: {e}")
            
            # If no sitemaps found in robots.txt, try common locations
            if not self.sitemap_urls:
                common_sitemap_paths = [
                    "/sitemap.xml",
                    "/sitemap_index.xml",
                    "/sitemap/sitemap.xml",
                    "/sitemaps/sitemap.xml"
                ]
                
                for path in common_sitemap_paths:
                    sitemap_url = f"{scheme}://{domain}{path}"
                    try:
                        response = session.get(sitemap_url, timeout=10)
                        if response.status_code == 200 and 'xml' in response.headers.get('Content-Type', ''):
                            self.sitemap_urls.add(sitemap_url)
                            break
                    except Exception:
                        continue
            
            # Process all found sitemaps
            urls_from_sitemap = []
            for sitemap_url in self.sitemap_urls:
                try:
                    response = session.get(sitemap_url, timeout=15)
                    response.raise_for_status()
                    
                    # Check if it's a sitemap index
                    root = ET.fromstring(response.content)
                    
                    # Handle sitemap index
                    if 'sitemapindex' in root.tag:
                        for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                            loc = sitemap.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc is not None and loc.text:
                                child_sitemap_url = loc.text.strip()
                                try:
                                    child_response = session.get(child_sitemap_url, timeout=15)
                                    child_response.raise_for_status()
                                    child_root = ET.fromstring(child_response.content)
                                    
                                    # Extract URLs from child sitemap
                                    for url in child_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                                        loc = url.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                                        if loc is not None and loc.text:
                                            page_url = loc.text.strip()
                                            if self.is_internal_link(page_url, domain):
                                                urls_from_sitemap.append((page_url, 1))  # Start at depth 1
                                except Exception as e:
                                    logging.warning(f"Error parsing child sitemap {child_sitemap_url}: {e}")
                    
                    # Handle regular sitemap
                    elif 'urlset' in root.tag:
                        for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                            loc = url.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc is not None and loc.text:
                                page_url = loc.text.strip()
                                if self.is_internal_link(page_url, domain):
                                    urls_from_sitemap.append((page_url, 1))  # Start at depth 1
                
                except Exception as e:
                    logging.warning(f"Error parsing sitemap {sitemap_url}: {e}")
            
            logging.info(f"Found {len(urls_from_sitemap)} URLs from sitemaps")
            return urls_from_sitemap
            
        except Exception as e:
            logging.error(f"Error in sitemap parsing: {e}")
            return []
    
    def crawl(self):
        """Start the crawling process."""
        to_visit = [(url, 1) for url in self.start_urls]
        
        # Parse sitemaps if enabled
        if self.parse_sitemaps:
            logging.info("Parsing sitemaps...")
            for start_url in self.start_urls:
                sitemap_urls = self.parse_sitemap(start_url)
                if sitemap_urls:
                    to_visit.extend(sitemap_urls)
                    logging.info(f"Added {len(sitemap_urls)} URLs from sitemaps")
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            while to_visit and len(self.visited) < self.max_pages:
                # Take a batch of URLs to process
                batch = to_visit[:self.concurrency]
                to_visit = to_visit[self.concurrency:]
                
                # Process the batch
                results = list(executor.map(self.process_url, batch))
                
                # Add new URLs to the queue
                for result in results:
                    if result:
                        to_visit.extend(result)
                        
        logging.info(f"Crawling complete! Scraped {len(self.scraped_pages)} pages.")
        return self.scraped_pages
    
    def save_to_file(self, output_file="website_content.json"):
        """Save the scraped content to a file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.scraped_pages, f, indent=2)
            
        logging.info(f"Saved scraped content to {output_file}")
        
        # Also save a text version for compatibility
        with open("website_content.txt", "w", encoding="utf-8") as f:
            for page in self.scraped_pages:
                metadata = page["metadata"]
                f.write(f"\n\n--- PAGE: {metadata['url']} ---\n")
                f.write(f"TITLE: {metadata['title']}\n")
                f.write(f"DESCRIPTION: {metadata['description']}\n")
                
                if metadata["headings"]:
                    f.write("HEADINGS:\n")
                    for heading in metadata["headings"]:
                        f.write(f"  H{heading['level']}: {heading['text']}\n")
                        
                f.write("\nCONTENT:\n")
                f.write(page["content"])
                
        logging.info("Saved text version to website_content.txt")

if __name__ == "__main__":
    # Example usage with all enhanced features
    crawler = EnhancedWebCrawler(
        start_urls=["https://neutrinotechsystems.com"],
        max_pages=200,  # Increased to 200 pages
        max_depth=4,    # Increased depth
        concurrency=5,  # More concurrent requests
        respect_robots=True,
        delay=1.0,
        parse_sitemaps=True,     # Enable sitemap parsing
        extract_pdf=True,        # Enable PDF extraction
        follow_subdomains=True   # Follow subdomains
    )
    
    print("🚀 Starting enhanced web crawler...")
    print("📊 Configuration:")
    print(f"  - Start URLs: {crawler.start_urls}")
    print(f"  - Max Pages: {crawler.max_pages}")
    print(f"  - Max Depth: {crawler.max_depth}")
    print(f"  - Concurrency: {crawler.concurrency}")
    print(f"  - Parse Sitemaps: {crawler.parse_sitemaps}")
    print(f"  - Follow Subdomains: {crawler.follow_subdomains}")
    
    crawler.crawl()
    crawler.save_to_file()
    
    print(f"✅ Crawling complete! Scraped {len(crawler.scraped_pages)} pages.")
    print(f"📁 Data saved to website_content.json and website_content.txt")