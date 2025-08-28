def monitor_cybersecurity_incidents(self, days_back: int = 7, use_rss: bool = True) -> List[CyberIncident]:
        """Main monitoring function with real SEC data and comprehensive logging"""
        logger.info(f"=== Starting cybersecurity monitoring ===")
        logger.info(f"Monitoring period: {days_back} days")
        logger.info(f"Data source: {'RSS' if use_rss else 'Search API'}")
        logger.info(f"Debug mode: {self.debug_mode}")
        
        # Clear previous API responses
        self.api_responses = []
        
        # Get recent 8-K filings
        if use_rss:
            logger.info("Using RSS feed method")
            filings = self.get_recent_8k_filings_rss(days_back)
        else:
            logger.info("Using Search API method")
            filings = self.get_recent_8k_filings_search(days_back)
        
        if not filings:
            logger.error("❌ No filings retrieved from SEC")
            return []
        
        logger.info(f"✅ Retrieved {len(filings)} filings for analysis")
        
        incidents = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, filing in enumerate(filings):
            progress = (i + 1) / len(filings)
            progress_bar.progress(progress)
            status_text.text(f'✅ Analyzed {len(filings)} filings, found {len(incidents)} incidents')
        
        # Sort by confidence score
        incidents.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return incidentsAnalyzing {filing["company_name"]} ({i+1}/{len(filings)})')
            
            logger.info(f"--- Analyzing filing {i+1}/{len(filings)}: {filing['company_name']} ---")
            
            # Download and analyze filing content
            content = self.analyze_8k_content(filing['filing_url'])
            
            if content:
                logger.info(f"Successfully downloaded content ({len(content)} characters)")
                incident = self.detect_cybersecurity_incident(content, filing)
                if incident:
                    incidents.append(incident)
                    logger.info(f"🚨 CYBERSECURITY INCIDENT DETECTED: {incident.company_name} (confidence: {incident.confidence_score:.2f})")
                else:
                    logger.info("No cybersecurity incident detected in this filing")
            else:
                logger.warning(f"Failed to download content for {filing['company_name']}")
        
        progress_bar.progress(1.0)
        logger.info(f"=== Monitoring Complete ===")
        logger.info(f"Total filings analyzed: {len(filings)}")
        logger.info(f"Cybersecurity incidents found: {len(incidents)}")
        
        status_text.text(f'import streamlit as st
import requests
import json
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
import plotly.express as px
import feedparser
from urllib.parse import urljoin
import logging
import traceback
from io import StringIO

# Custom logging handler for Streamlit
class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_buffer = StringIO()
    
    def emit(self, record):
        log_entry = self.format(record)
        self.log_buffer.write(f"{log_entry}\n")
        
    def get_logs(self):
        return self.log_buffer.getvalue()
    
    def clear_logs(self):
        self.log_buffer = StringIO()

# Configure logging with custom handler
streamlit_handler = StreamlitLogHandler()
streamlit_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(streamlit_handler)

# Configure Streamlit page
st.set_page_config(
    page_title="SEC 8-K Cybersecurity Monitor - Production",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class CyberIncident:
    """Data class for cybersecurity incident from 8-K filing"""
    company_name: str
    ticker: str
    cik: str
    filing_date: str
    filing_url: str
    incident_description: str
    incident_date: Optional[str]
    keywords: List[str]
    raw_content_preview: str
    filing_type: str
    confidence_score: float  # 0-1 score for incident detection confidence

@dataclass
class APIResponse:
    """Track SEC API responses for debugging"""
    url: str
    status_code: int
    response_time: float
    content_length: int
    error: Optional[str]
    timestamp: datetime

class ProductionEdgarMonitor:
    """Production SEC EDGAR monitor with real API calls and comprehensive debugging"""
    
    def __init__(self, company_name: str = "CyberIncidentTracker", email: str = "contact@company.com", debug_mode: bool = False):
        # SEC requires proper User-Agent with company name and email
        self.headers = {
            "User-Agent": f"{company_name} {email}",
            "Accept": "application/json, text/html, application/xml",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive"
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Debug mode settings
        self.debug_mode = debug_mode
        self.api_responses = []  # Track all API calls
        
        logger.info(f"Initialized EdgearMonitor with User-Agent: {self.headers['User-Agent']}")
        
        # SEC EDGAR endpoints
        self.edgar_base = "https://www.sec.gov"
        self.edgar_search_base = "https://efts.sec.gov/LATEST/search-index"
        self.edgar_rss = "https://www.sec.gov/cgi-bin/browse-edgar"
        
        # Alternative RSS endpoint
        self.edgar_rss_atom = "https://www.sec.gov/Archives/edgar/usgaap.rss.xml"
        
        logger.info(f"Base URLs configured - RSS: {self.edgar_rss}, Search: {self.edgar_search_base}")
        
        # Cybersecurity detection keywords (expanded for production)
        self.cyber_keywords = {
            'primary': [
                'cybersecurity incident', 'cyber attack', 'cyberattack', 'data breach', 
                'ransomware', 'malware', 'phishing', 'security incident', 'cyber incident'
            ],
            'secondary': [
                'unauthorized access', 'unauthorized entry', 'security breach', 
                'data compromise', 'system intrusion', 'network intrusion',
                'cyber threat', 'hacking', 'data theft', 'information security'
            ],
            'technical': [
                'ddos', 'denial of service', 'sql injection', 'zero-day',
                'advanced persistent threat', 'apt', 'trojan', 'backdoor',
                'vulnerability', 'exploitation', 'penetration'
            ]
        }
        
        # Rate limiting parameters
        self.request_delay = 0.1  # 100ms between requests (SEC allows 10/second)
        self.max_retries = 3
        
        logger.info(f"Initialized with {sum(len(v) for v in self.cyber_keywords.values())} keywords across {len(self.cyber_keywords)} categories")
    
    def _make_request(self, url: str, params: Dict = None, method: str = "GET") -> Optional[requests.Response]:
        """Make rate-limited request to SEC with comprehensive logging"""
        start_time = time.time()
        
        logger.debug(f"Making {method} request to: {url}")
        if params:
            logger.debug(f"Parameters: {params}")
        
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=30)
                else:
                    response = self.session.post(url, data=params, timeout=30)
                
                response_time = time.time() - start_time
                content_length = len(response.content) if response.content else 0
                
                # Log API response
                api_response = APIResponse(
                    url=url,
                    status_code=response.status_code,
                    response_time=response_time,
                    content_length=content_length,
                    error=None,
                    timestamp=datetime.now()
                )
                self.api_responses.append(api_response)
                
                logger.info(f"SEC API Response: {response.status_code} - {content_length} bytes - {response_time:.2f}s")
                
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                error_msg = f"Request attempt {attempt + 1} failed for {url}: {str(e)}"
                logger.warning(error_msg)
                
                # Log failed API response
                api_response = APIResponse(
                    url=url,
                    status_code=getattr(e.response, 'status_code', 0) if hasattr(e, 'response') and e.response else 0,
                    response_time=time.time() - start_time,
                    content_length=0,
                    error=str(e),
                    timestamp=datetime.now()
                )
                self.api_responses.append(api_response)
                
                if attempt == self.max_retries - 1:
                    logger.error(f"All requests failed for {url}: {str(e)}")
                    if self.debug_mode:
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
                
        return None
    
    def get_recent_8k_filings_rss(self, days_back: int = 7) -> List[Dict]:
        """Get recent 8-K filings using SEC RSS feed with enhanced debugging"""
        logger.info(f"Starting RSS feed fetch for last {days_back} days")
        
        # Try multiple RSS endpoints
        rss_endpoints = [
            {
                'url': self.edgar_rss,
                'params': {
                    'action': 'getcurrent',
                    'type': '8-K',
                    'output': 'atom',
                    'count': min(100, days_back * 20)
                },
                'name': 'Main RSS Feed'
            },
            {
                'url': f"{self.edgar_base}/Archives/edgar/monthly/xbrlrss-2024-08.xml",
                'params': {},
                'name': 'Monthly XBRL RSS'
            }
        ]
        
        for endpoint in rss_endpoints:
            logger.info(f"Trying {endpoint['name']}: {endpoint['url']}")
            
            response = self._make_request(endpoint['url'], endpoint['params'])
            if not response:
                logger.warning(f"Failed to get response from {endpoint['name']}")
                continue
            
            logger.info(f"Received {len(response.content)} bytes from {endpoint['name']}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Log first 500 chars of response for debugging
            content_preview = response.text[:500] if response.text else "No text content"
            logger.debug(f"Response preview: {content_preview}")
            
            try:
                # Try feedparser first (more robust)
                logger.info("Attempting to parse with feedparser...")
                feed = feedparser.parse(response.content)
                
                if hasattr(feed, 'entries') and len(feed.entries) > 0:
                    logger.info(f"Feedparser found {len(feed.entries)} entries")
                    return self._process_rss_entries(feed.entries, days_back)
                else:
                    logger.warning(f"Feedparser found no entries. Feed info: {getattr(feed, 'feed', {})}")
                
                # Try XML parsing as backup
                logger.info("Attempting manual XML parsing...")
                filings = self._parse_rss_xml_manually(response.content, days_back)
                if filings:
                    return filings
                
            except Exception as e:
                logger.error(f"Error parsing RSS from {endpoint['name']}: {str(e)}")
                if self.debug_mode:
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
        
        logger.error("All RSS endpoints failed")
        return []
    
    def _process_rss_entries(self, entries, days_back: int) -> List[Dict]:
        """Process RSS entries into filing data"""
        logger.info(f"Processing {len(entries)} RSS entries")
        
        filings = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for i, entry in enumerate(entries):
            try:
                # Log entry details
                logger.debug(f"Entry {i+1}: {getattr(entry, 'title', 'No title')}")
                
                # Parse entry date
                entry_date = None
                for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
                    if hasattr(entry, date_field) and getattr(entry, date_field):
                        try:
                            entry_date = datetime(*getattr(entry, date_field)[:6])
                            break
                        except:
                            continue
                
                if not entry_date:
                    logger.debug(f"Could not parse date for entry: {entry.title}")
                    continue
                
                if entry_date < cutoff_date:
                    logger.debug(f"Entry too old: {entry_date} < {cutoff_date}")
                    continue
                
                # Extract company info from title
                title = getattr(entry, 'title', '')
                if '8-K' not in title.upper():
                    logger.debug(f"Not an 8-K filing: {title}")
                    continue
                
                company_match = re.search(r'8-K\s*-\s*(.+?)\s*\(([^)]+)\)', title, re.IGNORECASE)
                if company_match:
                    company_name = company_match.group(1).strip()
                    cik_info = company_match.group(2)
                    cik_match = re.search(r'\b(\d{10})\b', cik_info)
                    cik = cik_match.group(1) if cik_match else ""
                    
                    filing = {
                        'title': title,
                        'company_name': company_name,
                        'cik': cik,
                        'filing_url': getattr(entry, 'link', ''),
                        'published': entry_date.isoformat(),
                        'summary': getattr(entry, 'summary', '')
                    }
                    filings.append(filing)
                    logger.debug(f"Added filing: {company_name} ({cik})")
                else:
                    logger.debug(f"Could not parse company from title: {title}")
                    
            except Exception as e:
                logger.error(f"Error processing entry {i+1}: {str(e)}")
                if self.debug_mode:
                    logger.error(f"Entry data: {entry}")
        
        logger.info(f"Processed {len(filings)} valid 8-K filings")
        return filings
    
    def _parse_rss_xml_manually(self, xml_content: bytes, days_back: int) -> List[Dict]:
        """Manual XML parsing as backup method"""
        logger.info("Attempting manual XML parsing")
        
        try:
            root = ET.fromstring(xml_content)
            logger.info(f"XML root tag: {root.tag}")
            
            # Find all entry/item elements
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry') or root.findall('.//item')
            logger.info(f"Found {len(entries)} XML entries")
            
            if not entries:
                # Log XML structure for debugging
                logger.debug("XML structure:")
                for child in root:
                    logger.debug(f"  {child.tag}: {child.text[:100] if child.text else 'No text'}")
            
            # Process entries similar to RSS processing
            # Implementation would depend on actual XML structure
            return []
            
        except ET.ParseError as e:
            logger.error(f"XML parsing failed: {str(e)}")
            return []
    
    def get_recent_8k_filings_search(self, days_back: int = 7) -> List[Dict]:
        """Alternative method using SEC search API"""
        logger.info(f"Fetching 8-K filings via search API")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # SEC search parameters
        search_params = {
            'dateRange': 'custom',
            'startdt': start_date.strftime('%Y-%m-%d'),
            'enddt': end_date.strftime('%Y-%m-%d'),
            'forms': '8-K',
            'from': 0,
            'size': 100
        }
        
        response = self._make_request(self.edgar_search_base, search_params)
        if not response:
            return []
        
        try:
            data = response.json()
            filings = []
            
            for hit in data.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                
                filing = {
                    'title': f"8-K - {source.get('display_names', ['Unknown'])[0]}",
                    'company_name': source.get('display_names', ['Unknown'])[0],
                    'cik': source.get('ciks', [''])[0],
                    'filing_url': f"{self.edgar_base}/Archives/edgar/data/{source.get('ciks', [''])[0]}/{source.get('file_num', '')}.htm",
                    'published': source.get('file_date', ''),
                    'period_ending': source.get('period_ending', '')
                }
                filings.append(filing)
            
            return filings
            
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return []
    
    def analyze_8k_content(self, filing_url: str) -> Optional[str]:
        """Download and extract text content from 8-K filing with detailed logging"""
        logger.info(f"Analyzing filing content: {filing_url}")
        
        # Validate URL
        if not filing_url or not filing_url.startswith('http'):
            logger.error(f"Invalid filing URL: {filing_url}")
            return None
        
        response = self._make_request(filing_url)
        if not response:
            logger.error(f"Failed to fetch filing content from {filing_url}")
            return None
        
        try:
            # Check content type
            content_type = response.headers.get('content-type', '')
            logger.info(f"Content-Type: {content_type}")
            
            # Log response size
            content_size = len(response.content)
            logger.info(f"Downloaded {content_size} bytes")
            
            if content_size == 0:
                logger.warning("Empty response content")
                return None
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"Extracted {len(cleaned_text)} characters of text")
            
            # Log preview of content for debugging
            preview = cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
            logger.debug(f"Content preview: {preview}")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error parsing filing content: {str(e)}")
            if self.debug_mode:
                logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def detect_cybersecurity_incident(self, content: str, filing_info: Dict) -> Optional[CyberIncident]:
        """Analyze filing content for cybersecurity incidents with confidence scoring"""
        if not content:
            return None
        
        content_lower = content.lower()
        
        # Check for Item 1.05 (cybersecurity incidents)
        item_105_patterns = [
            r'item\s+1\.0?5[^0-9]',
            r'item\s+1\s+0\s*5[^0-9]',
            r'cybersecurity\s+incident',
            r'material\s+cybersecurity'
        ]
        
        has_item_105 = any(re.search(pattern, content_lower) for pattern in item_105_patterns)
        
        if not has_item_105:
            return None
        
        # Score cybersecurity keywords
        found_keywords = []
        confidence_score = 0.0
        
        # Primary keywords (high confidence)
        for keyword in self.cyber_keywords['primary']:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
                confidence_score += 0.3
        
        # Secondary keywords (medium confidence)
        for keyword in self.cyber_keywords['secondary']:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
                confidence_score += 0.2
        
        # Technical keywords (lower confidence but specific)
        for keyword in self.cyber_keywords['technical']:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
                confidence_score += 0.1
        
        # Minimum confidence threshold
        if confidence_score < 0.3 or not found_keywords:
            return None
        
        # Extract incident description (Item 1.05 section)
        incident_description = self._extract_item_105_section(content)
        
        # Extract incident date
        incident_date = self._extract_incident_date(incident_description)
        
        # Extract company details
        ticker = self._extract_ticker_from_content(content)
        
        return CyberIncident(
            company_name=filing_info.get('company_name', 'Unknown'),
            ticker=ticker,
            cik=filing_info.get('cik', ''),
            filing_date=filing_info.get('published', '')[:10],
            filing_url=filing_info.get('filing_url', ''),
            incident_description=incident_description[:1000],  # Truncate for display
            incident_date=incident_date,
            keywords=list(set(found_keywords)),  # Remove duplicates
            raw_content_preview=content[:500],
            filing_type='8-K',
            confidence_score=min(confidence_score, 1.0)
        )
    
    def _extract_item_105_section(self, content: str) -> str:
        """Extract Item 1.05 section from filing"""
        # Find Item 1.05 section
        patterns = [
            r'item\s+1\.0?5.*?(?=item\s+[2-9]|item\s+10|signature|$)',
            r'cybersecurity\s+incident.*?(?=item|signature|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                section = match.group(0)
                # Clean and truncate
                section = re.sub(r'\s+', ' ', section).strip()
                return section[:2000]  # Limit length
        
        # Fallback: look for cybersecurity-related paragraphs
        paragraphs = content.split('\n\n')
        cyber_paragraphs = []
        
        for para in paragraphs:
            if any(keyword in para.lower() for keyword in self.cyber_keywords['primary']):
                cyber_paragraphs.append(para.strip())
        
        return ' '.join(cyber_paragraphs)[:2000] if cyber_paragraphs else "Description not extracted"
    
    def _extract_incident_date(self, description: str) -> Optional[str]:
        """Extract incident date from description"""
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'on\s+(\w+\s+\d{1,2},?\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def get_api_communication_log(self) -> pd.DataFrame:
        """Return detailed log of all SEC API communications"""
        if not self.api_responses:
            return pd.DataFrame()
        
        log_data = []
        for response in self.api_responses:
            log_data.append({
                'Timestamp': response.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'URL': response.url[:50] + '...' if len(response.url) > 50 else response.url,
                'Status Code': response.status_code,
                'Response Time (s)': round(response.response_time, 2),
                'Content Length': response.content_length,
                'Error': response.error or 'Success',
                'Full URL': response.url  # Hidden column for reference
            })
        
        return pd.DataFrame(log_data)
    
    def get_debug_summary(self) -> Dict:
        """Get comprehensive debug information"""
        return {
            'total_api_calls': len(self.api_responses),
            'successful_calls': len([r for r in self.api_responses if r.status_code == 200]),
            'failed_calls': len([r for r in self.api_responses if r.status_code != 200]),
            'total_data_downloaded': sum(r.content_length for r in self.api_responses),
            'average_response_time': sum(r.response_time for r in self.api_responses) / len(self.api_responses) if self.api_responses else 0,
            'user_agent': self.headers.get('User-Agent'),
            'rate_limit_delay': self.request_delay,
            'max_retries': self.max_retries,
            'debug_mode': self.debug_mode
        }
    
    def _extract_ticker_from_content(self, content: str) -> str:
        """Extract ticker symbol from filing content"""
        patterns = [
            r'trading\s+symbol[:\s]*([A-Z]{1,5})',
            r'ticker[:\s]*([A-Z]{1,5})',
            r'nasdaq[:\s]*([A-Z]{1,5})',
            r'nyse[:\s]*([A-Z]{1,5})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
        """Main monitoring function with real SEC data"""
        logger.info(f"Starting cybersecurity monitoring for {days_back} days")
        
        # Get recent 8-K filings
        if use_rss:
            filings = self.get_recent_8k_filings_rss(days_back)
        else:
            filings = self.get_recent_8k_filings_search(days_back)
        
        if not filings:
            logger.warning("No filings retrieved")
            return []
        
        incidents = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, filing in enumerate(filings):
            progress = (i + 1) / len(filings)
            progress_bar.progress(progress)
            status_text.text(f'Analyzing {filing["company_name"]} ({i+1}/{len(filings)})')
            
            # Download and analyze filing content
            content = self.analyze_8k_content(filing['filing_url'])
            
            if content:
                incident = self.detect_cybersecurity_incident(content, filing)
                if incident:
                    incidents.append(incident)
                    logger.info(f"Cybersecurity incident detected: {incident.company_name}")
        
        progress_bar.progress(1.0)
        status_text.text(f'✅ Analyzed {len(filings)} filings, found {len(incidents)} incidents')
        
        # Sort by confidence score
        incidents.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return incidents

def main():
    """Production Streamlit app with real SEC data and comprehensive debugging"""
    
    st.title("🔍 SEC 8-K Cybersecurity Monitor - Production")
    st.markdown("**Real-time monitoring of cybersecurity incidents using live SEC EDGAR data**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox(
        "🐛 Enable Debug Mode",
        value=False,
        help="Show detailed API communication logs and error traces"
    )
    
    # User contact info (required by SEC)
    company_name = st.sidebar.text_input(
        "Company/Organization Name",
        value="CyberSecurityTracker",
        help="Required by SEC for API access"
    )
    
    email = st.sidebar.text_input(
        "Contact Email", 
        value="contact@company.com",
        help="Required by SEC for API access"
    )
    
    if not company_name or not email or '@' not in email:
        st.error("⚠️ SEC requires valid company name and email for API access")
        return
    
    days_back = st.sidebar.slider(
        "Monitoring Period (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="How many days back to scan"
    )
    
    use_rss = st.sidebar.radio(
        "Data Source",
        ["RSS Feed", "Search API"],
        help="RSS is more reliable, Search API provides more metadata"
    ) == "RSS Feed"
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence Score",
        min_value=0.1,
        max_value=1.0,
        value=0.3,  # Lowered for better debugging
        step=0.1,
        help="Filter incidents by detection confidence"
    )
    
    # Debug information in sidebar
    if debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🐛 Debug Info**")
        st.sidebar.markdown(f"User-Agent will be: `{company_name} {email}`")
        st.sidebar.markdown(f"Rate limit: 0.1s between requests")
        st.sidebar.markdown(f"Max retries: 3 per request")
    
    # Main monitoring
    st.header("📊 Live Incident Detection")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Scan live SEC EDGAR filings for cybersecurity incidents")
    with col2:
        run_monitor = st.button("🔍 Start Scan", type="primary")
    
    if run_monitor:
        # Clear previous logs
        if hasattr(streamlit_handler, 'clear_logs'):
            streamlit_handler.clear_logs()
        
        # Initialize production monitor
        with st.spinner("Initializing SEC EDGAR connection..."):
            monitor = ProductionEdgarMonitor(company_name, email, debug_mode=debug_mode)
        
        # Debug information display
        if debug_mode:
            with st.expander("🐛 Debug Information", expanded=True):
                debug_info = monitor.get_debug_summary()
                st.json(debug_info)
        
        # Run monitoring
        with st.spinner("Scanning SEC filings..."):
            incidents = monitor.monitor_cybersecurity_incidents(days_back, use_rss)
        
        # Show API communication log
        if debug_mode:
            st.header("🔗 SEC API Communication Log")
            api_log = monitor.get_api_communication_log()
            if not api_log.empty:
                st.dataframe(api_log.drop('Full URL', axis=1), use_container_width=True)
                
                # API Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total API Calls", len(api_log))
                with col2:
                    success_rate = len(api_log[api_log['Status Code'] == 200]) / len(api_log) * 100 if len(api_log) > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with col3:
                    avg_time = api_log['Response Time (s)'].mean() if len(api_log) > 0 else 0
                    st.metric("Avg Response Time", f"{avg_time:.2f}s")
                with col4:
                    total_data = api_log['Content Length'].sum()
                    st.metric("Total Data", f"{total_data:,} bytes")
            else:
                st.warning("No API calls recorded")
        
        # Show system logs
        if debug_mode:
            st.header("📋 System Logs")
            logs = streamlit_handler.get_logs()
            if logs:
                st.text_area("Debug Logs", logs, height=300)
            else:
                st.info("No logs generated")
        
        # Filter by confidence
        high_confidence_incidents = [i for i in incidents if i.confidence_score >= min_confidence]
        all_incidents_count = len(incidents)
        
        # Results summary
        if incidents:
            st.success(f"Scan complete! Found {len(high_confidence_incidents)} high-confidence incidents out of {all_incidents_count} total detections.")
        else:
            st.warning("Scan complete! No cybersecurity incidents detected.")
            if debug_mode:
                st.info("Check the debug logs above to see what happened during the scan.")
        
        if high_confidence_incidents:
            # Display results
            st.header("🚨 Cybersecurity Incidents Detected")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("High Confidence", len(high_confidence_incidents))
            with col2:
                st.metric("Total Detections", all_incidents_count)
            with col3:
                avg_confidence = sum(i.confidence_score for i in high_confidence_incidents) / len(high_confidence_incidents)
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            with col4:
                st.metric("Companies Affected", len(set(i.company_name for i in high_confidence_incidents)))
            
            # Incidents table
            incidents_data = []
            for incident in high_confidence_incidents:
                incidents_data.append({
                    "Company": incident.company_name,
                    "Ticker": incident.ticker or "N/A",
                    "Filing Date": incident.filing_date,
                    "Confidence": f"{incident.confidence_score:.2f}",
                    "Keywords": len(incident.keywords),
                    "Top Keywords": ", ".join(incident.keywords[:3])
                })
            
            df = pd.DataFrame(incidents_data)
            st.dataframe(df, use_container_width=True)
            
            # Detailed views
            for i, incident in enumerate(high_confidence_incidents):
                with st.expander(f"#{i+1}: {incident.company_name} - Confidence: {incident.confidence_score:.2f}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**📝 Incident Description:**")
                        st.write(incident.incident_description)
                        
                        st.markdown("**🔗 Original SEC Filing:**")
                        st.link_button("View 8-K Filing", incident.filing_url)
                        
                        if debug_mode:
                            st.markdown("**🐛 Raw Content Preview:**")
                            st.text_area(f"Raw content preview #{i+1}", incident.raw_content_preview, height=100)
                    
                    with col2:
                        st.json({
                            "Company": incident.company_name,
                            "Ticker": incident.ticker,
                            "CIK": incident.cik,
                            "Filing Date": incident.filing_date,
                            "Incident Date": incident.incident_date,
                            "Confidence Score": incident.confidence_score,
                            "Keywords": incident.keywords
                        })
        
        elif all_incidents_count > 0:
            st.info(f"Found {all_incidents_count} potential incidents, but none met the confidence threshold of {min_confidence:.1f}")
            st.markdown("**Try lowering the confidence threshold in the sidebar to see more results.**")
            
            # Show low-confidence incidents in debug mode
            if debug_mode:
                st.subheader("🔍 Low-Confidence Detections (Debug)")
                low_conf_incidents = [i for i in incidents if i.confidence_score < min_confidence]
                for i, incident in enumerate(low_conf_incidents):
                    with st.expander(f"Low-conf #{i+1}: {incident.company_name} - Confidence: {incident.confidence_score:.2f}"):
                        st.write(f"**Keywords:** {', '.join(incident.keywords)}")
                        st.write(f"**Description:** {incident.incident_description[:200]}...")
        
        else:
            st.info(f"No cybersecurity incidents found in the last {days_back} days.")
            if debug_mode:
                st.markdown("**Troubleshooting tips:**")
                st.markdown("- Check the API Communication Log above for any failed requests")
                st.markdown("- Verify your company name and email are valid")
                st.markdown("- Try increasing the monitoring period")
                st.markdown("- Check if SEC EDGAR is accessible from your location")
    
    # Production notes
    st.header("🏭 Production Features")
    st.markdown("""
    **This production version includes:**
    - ✅ Real SEC EDGAR API integration
    - ✅ Multiple data source options (RSS + Search API)
    - ✅ Confidence scoring for incident detection
    - ✅ Proper SEC rate limiting and compliance
    - ✅ Advanced keyword detection algorithms
    - ✅ Robust error handling and retries
    - ✅ Comprehensive debugging and logging
    - 🆕 **Debug mode with SEC API communication logs**
    - 🆕 **Detailed error tracking and system logs**
    
    **Rate Limits:** Respects SEC's 10 requests/second limit
    **Data Sources:** Live SEC EDGAR RSS feeds and search API
    **Detection:** Multi-tier keyword analysis with confidence scoring
    """)
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting Guide"):
        st.markdown("""
        **If you're getting no results:**
        
        1. **Enable Debug Mode** - Toggle the debug switch in the sidebar to see detailed logs
        2. **Check API Communication** - Look for failed requests in the debug section
        3. **Verify Credentials** - Make sure company name and email are valid
        4. **Try Different Settings**:
           - Increase monitoring period to 14-30 days
           - Lower confidence threshold to 0.2-0.3
           - Switch between RSS Feed and Search API
        5. **Check Recent Activity** - There may simply be no recent cybersecurity incidents
        
        **Common Issues:**
        - **Empty API responses**: SEC servers may be temporarily unavailable
        - **Parsing errors**: Content format may have changed
        - **Rate limiting**: Built-in delays should prevent this
        - **Network issues**: Check your internet connection
        """)

if __name__ == "__main__":
    main()
