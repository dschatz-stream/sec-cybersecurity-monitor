# SEC 8-K Cybersecurity Monitor - Bulk Data Version
# Version: 2.0.0
# Features: Bulk data collection, 1000+ filings, daily index parsing
# Last Updated: 2024-08-28

import streamlit as st
import requests
import json
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import logging
import traceback
from io import StringIO, BytesIO
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# Version information
VERSION = "2.0.0"
BUILD_DATE = "2024-08-28"
FEATURES = [
    "SEC Bulk Data API Integration",
    "Daily Index File Processing", 
    "1000+ Filing Capability",
    "Multi-threaded Analysis",
    "Company Ticker Mapping",
    "Enhanced Debug Logging"
]

# Try to import dateutil, fallback if not available
try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

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
    page_title="SEC 8-K Monitor - Bulk Data Version",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class CyberIncident:
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
    confidence_score: float

@dataclass
class APIResponse:
    url: str
    status_code: int
    response_time: float
    content_length: int
    error: Optional[str]
    timestamp: datetime

class SECBulkDataMonitor:
    """SEC Monitor using bulk data APIs to get 1000+ filings"""
    
    def __init__(self, company_name: str = "CyberIncidentTracker", email: str = "contact@company.com", debug_mode: bool = False):
        self.headers = {
            "User-Agent": f"{company_name} {email}",
            "Accept": "application/json, text/html, application/xml",
            "Accept-Encoding": "gzip, deflate"
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.debug_mode = debug_mode
        self.api_responses = []
        
        # SEC bulk data endpoints
        self.sec_base = "https://www.sec.gov"
        self.submissions_api = "https://data.sec.gov/submissions"  # Company submissions API
        self.bulk_data_base = "https://www.sec.gov/Archives/edgar/daily-index"
        self.company_tickers_url = "https://www.sec.gov/files/company_tickers.json"
        
        logger.info(f"Initialized SEC Bulk Data Monitor")
        logger.info(f"User-Agent: {self.headers['User-Agent']}")
        
        # Cybersecurity keywords
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
        
        self.request_delay = 0.1
        self.max_retries = 3
        
    def _make_request(self, url: str, params: Dict = None, method: str = "GET") -> Optional[requests.Response]:
        """Make rate-limited request with logging"""
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
                    return None
                time.sleep(2 ** attempt)
                
        return None
    
    def get_company_tickers_mapping(self) -> Dict[str, str]:
        """Get mapping of company tickers to CIKs"""
        logger.info("Fetching company ticker to CIK mapping...")
        
        response = self._make_request(self.company_tickers_url)
        if not response:
            logger.warning("Failed to get company ticker mapping")
            return {}
        
        try:
            data = response.json()
            # Convert to CIK -> ticker mapping
            cik_to_ticker = {}
            for ticker_info in data.values():
                if isinstance(ticker_info, dict):
                    cik = str(ticker_info.get('cik_str', '')).zfill(10)
                    ticker = ticker_info.get('ticker', '')
                    if cik and ticker:
                        cik_to_ticker[cik] = ticker
            
            logger.info(f"Loaded {len(cik_to_ticker)} company ticker mappings")
            return cik_to_ticker
            
        except Exception as e:
            logger.error(f"Error parsing company tickers: {e}")
            return {}
    
    def get_bulk_8k_filings(self, days_back: int, max_filings: int = 1000) -> List[Dict]:
        """Get bulk 8-K filings using SEC's daily index files"""
        logger.info(f"Starting bulk 8-K filing retrieval for {days_back} days, max {max_filings} filings")
        
        all_filings = []
        cik_to_ticker = self.get_company_tickers_mapping()
        
        # Try daily index method first
        for day_offset in range(days_back):
            if len(all_filings) >= max_filings:
                break
                
            target_date = datetime.now() - timedelta(days=day_offset)
            
            # Skip weekends (SEC doesn't publish on weekends)
            if target_date.weekday() >= 5:
                continue
            
            logger.info(f"Fetching filings for {target_date.strftime('%Y-%m-%d')}...")
            
            daily_filings = self.get_daily_index_filings(target_date, cik_to_ticker)
            
            if daily_filings:
                all_filings.extend(daily_filings)
                logger.info(f"Found {len(daily_filings)} 8-K filings for {target_date.strftime('%Y-%m-%d')}")
            
            # Respect rate limits
            time.sleep(0.2)
        
        # If daily index method didn't work well, try fallback RSS method with higher limits
        if len(all_filings) < 50:  # If we got very few filings, try fallback
            logger.info("Daily index method returned few results, trying RSS fallback...")
            rss_filings = self.get_rss_fallback_filings(days_back, max_filings)
            if rss_filings:
                all_filings.extend(rss_filings)
                logger.info(f"RSS fallback found {len(rss_filings)} additional filings")
        
        # Remove duplicates and limit to max_filings
        seen = set()
        unique_filings = []
        for filing in all_filings:
            key = (filing.get('cik', ''), filing.get('title', ''))
            if key not in seen and len(unique_filings) < max_filings:
                seen.add(key)
                unique_filings.append(filing)
        
        logger.info(f"Total bulk filings retrieved: {len(unique_filings)}")
        return unique_filings
    
    def get_rss_fallback_filings(self, days_back: int, max_filings: int) -> List[Dict]:
        """Fallback RSS method if daily index files don't work"""
        logger.info("Using RSS fallback method...")
        
        try:
            import feedparser
            
            # Use the RSS endpoint
            rss_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                'action': 'getcurrent',
                'type': '8-K',
                'output': 'atom',
                'count': min(max_filings, 1000)
            }
            
            response = self._make_request(rss_url, params)
            if not response:
                return []
            
            # Parse RSS feed
            feed = feedparser.parse(response.content)
            filings = []
            
            if hasattr(feed, 'entries'):
                logger.info(f"RSS fallback found {len(feed.entries)} entries")
                
                cutoff_date = datetime.now() - timedelta(days=days_back)
                
                for entry in feed.entries:
                    # Parse entry date
                    entry_date = datetime.now()  # Default to now
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            entry_date = datetime(*entry.published_parsed[:6])
                        except:
                            pass
                    
                    if entry_date < cutoff_date:
                        continue
                    
                    # Extract company info
                    title = getattr(entry, 'title', '')
                    if '8-K' not in title.upper():
                        continue
                    
                    company_match = re.search(r'8-K\s*(?:/A)?\s*-\s*(.+?)\s*\(([^)]+)\)', title, re.IGNORECASE)
                    if company_match:
                        company_name = company_match.group(1).strip()
                        cik_info = company_match.group(2)
                        cik_match = re.search(r'\b(\d{10})\b', cik_info)
                        cik = cik_match.group(1) if cik_match else ""
                        
                        filing = {
                            'title': title,
                            'company_name': company_name,
                            'cik': cik,
                            'ticker': '',  # Will be filled later if needed
                            'filing_url': getattr(entry, 'link', ''),
                            'published': entry_date.isoformat(),
                            'form_type': '8-K',
                            'summary': getattr(entry, 'summary', '')
                        }
                        filings.append(filing)
            
            return filings
            
        except Exception as e:
            logger.error(f"RSS fallback failed: {e}")
            return []
    
    def get_daily_index_filings(self, target_date: datetime, cik_to_ticker: Dict[str, str]) -> List[Dict]:
        """Get 8-K filings from SEC daily index for specific date"""
        
        # SEC daily index URL format
        year = target_date.year
        quarter = f"QTR{(target_date.month - 1) // 3 + 1}"
        date_str = target_date.strftime('%Y%m%d')
        
        # Try both form.idx and master.idx formats
        index_urls = [
            f"{self.bulk_data_base}/{year}/{quarter}/form.{date_str}.idx",
            f"{self.bulk_data_base}/{year}/{quarter}/master.{date_str}.idx"
        ]
        
        for index_url in index_urls:
            logger.debug(f"Trying daily index: {index_url}")
            
            response = self._make_request(index_url)
            if response:
                filings = self.parse_daily_index(response.text, target_date, cik_to_ticker)
                if filings:
                    return filings
        
        logger.debug(f"No daily index found for {target_date.strftime('%Y-%m-%d')}")
        return []
    
    def parse_daily_index(self, index_content: str, filing_date: datetime, cik_to_ticker: Dict[str, str]) -> List[Dict]:
        """Parse SEC daily index file to extract 8-K filings"""
        
        filings = []
        lines = index_content.split('\n')
        
        logger.debug(f"Parsing index file with {len(lines)} lines")
        
        # Log first few lines to understand format
        if self.debug_mode and len(lines) > 10:
            logger.debug("First 10 lines of index file:")
            for i, line in enumerate(lines[:10]):
                logger.debug(f"Line {i}: {line}")
        
        # Look for different possible formats
        data_started = False
        found_8k_count = 0
        
        for line_num, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip header lines (usually first 10-15 lines)
            if line_num < 15 and any(header in line.upper() for header in ['FORM TYPE', 'COMPANY NAME', 'CIK', 'DATE FILED', 'FILE NAME', '----']):
                continue
                
            # Check if this looks like a data line
            # Format can be: pipe-delimited OR space-delimited OR tab-delimited
            
            # Try pipe-delimited first (most common)
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:
                    form_type = parts[0]
                    if form_type.upper() in ['8-K', '8-K/A']:
                        found_8k_count += 1
                        filing = self._parse_pipe_delimited_line(parts, filing_date, cik_to_ticker)
                        if filing:
                            filings.append(filing)
                            
            # Try space/tab delimited format
            elif any(form in line.upper() for form in ['8-K ', '8-K\t']):
                # Split on multiple spaces or tabs
                parts = re.split(r'\s{2,}|\t+', line)
                if len(parts) >= 4:
                    found_8k_count += 1
                    filing = self._parse_space_delimited_line(parts, filing_date, cik_to_ticker)
                    if filing:
                        filings.append(filing)
                        
            # Try fixed-width format (sometimes used)
            elif line.upper().startswith('8-K') or '8-K' in line[:20]:
                found_8k_count += 1
                filing = self._parse_fixed_width_line(line, filing_date, cik_to_ticker)
                if filing:
                    filings.append(filing)
        
        logger.debug(f"Found {found_8k_count} potential 8-K lines, parsed {len(filings)} valid filings")
        
        # If we found 8-K mentions but no valid filings, log some examples
        if found_8k_count > 0 and len(filings) == 0 and self.debug_mode:
            logger.debug("Found 8-K mentions but no valid filings. Sample lines with 8-K:")
            count = 0
            for line in lines:
                if '8-K' in line.upper() and count < 3:
                    logger.debug(f"Sample 8-K line: {line}")
                    count += 1
        
        return filings
    
    def _parse_pipe_delimited_line(self, parts: List[str], filing_date: datetime, cik_to_ticker: Dict[str, str]) -> Optional[Dict]:
        """Parse pipe-delimited format line"""
        try:
            if len(parts) >= 5:
                form_type = parts[0]
                company_name = parts[1]
                cik_raw = parts[2]
                date_filed = parts[3]
                filename = parts[4]
                
                # Clean CIK
                cik_clean = re.sub(r'[^\d]', '', cik_raw)
                if cik_clean and len(cik_clean) <= 10:
                    cik = cik_clean.zfill(10)
                    ticker = cik_to_ticker.get(cik, '')
                    
                    return {
                        'title': f"{form_type} - {company_name} ({cik}) (Filer)",
                        'company_name': company_name,
                        'cik': cik,
                        'ticker': ticker,
                        'filing_url': f"{self.sec_base}/Archives/{filename}",
                        'published': filing_date.isoformat(),
                        'form_type': form_type,
                        'filename': filename,
                        'date_filed': date_filed
                    }
        except Exception as e:
            logger.debug(f"Error parsing pipe-delimited line: {e}")
        return None
    
    def _parse_space_delimited_line(self, parts: List[str], filing_date: datetime, cik_to_ticker: Dict[str, str]) -> Optional[Dict]:
        """Parse space-delimited format line"""
        try:
            if len(parts) >= 4:
                form_type = parts[0]
                company_name = parts[1]
                cik_raw = parts[2] if len(parts) > 2 else ""
                filename = parts[-1]  # Usually last part
                
                # Clean CIK
                cik_clean = re.sub(r'[^\d]', '', cik_raw)
                if cik_clean and len(cik_clean) <= 10:
                    cik = cik_clean.zfill(10)
                    ticker = cik_to_ticker.get(cik, '')
                    
                    return {
                        'title': f"{form_type} - {company_name} ({cik}) (Filer)",
                        'company_name': company_name,
                        'cik': cik,
                        'ticker': ticker,
                        'filing_url': f"{self.sec_base}/Archives/{filename}",
                        'published': filing_date.isoformat(),
                        'form_type': form_type,
                        'filename': filename
                    }
        except Exception as e:
            logger.debug(f"Error parsing space-delimited line: {e}")
        return None
    
    def _parse_fixed_width_line(self, line: str, filing_date: datetime, cik_to_ticker: Dict[str, str]) -> Optional[Dict]:
        """Parse fixed-width format line"""
        try:
            # Common fixed-width format positions
            if len(line) > 80:
                form_type = line[:12].strip()
                company_name = line[12:62].strip()
                cik_raw = line[62:72].strip()
                filename = line[72:].strip()
                
                # Clean CIK
                cik_clean = re.sub(r'[^\d]', '', cik_raw)
                if cik_clean and len(cik_clean) <= 10:
                    cik = cik_clean.zfill(10)
                    ticker = cik_to_ticker.get(cik, '')
                    
                    return {
                        'title': f"{form_type} - {company_name} ({cik}) (Filer)",
                        'company_name': company_name,
                        'cik': cik,
                        'ticker': ticker,
                        'filing_url': f"{self.sec_base}/Archives/{filename}",
                        'published': filing_date.isoformat(),
                        'form_type': form_type,
                        'filename': filename
                    }
        except Exception as e:
            logger.debug(f"Error parsing fixed-width line: {e}")
        return None
    
    def analyze_8k_content(self, filing_url: str) -> Optional[str]:
        """Download and extract text content from 8-K filing"""
        logger.debug(f"Analyzing filing content: {filing_url}")
        
        if not filing_url or not filing_url.startswith('http'):
            logger.error(f"Invalid filing URL: {filing_url}")
            return None
        
        response = self._make_request(filing_url)
        if not response:
            return None
        
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "meta", "link"]):
                element.decompose()
            
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.debug(f"Extracted {len(cleaned_text)} characters")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error parsing filing content: {str(e)}")
            return None
    
    def detect_cybersecurity_incident(self, content: str, filing_info: Dict) -> Optional[CyberIncident]:
        """Analyze filing content for cybersecurity incidents"""
        if not content:
            return None
        
        content_lower = content.lower()
        
        # Check for Item 1.05 (cybersecurity incidents)
        item_105_patterns = [
            r'item\s+1\.0?5[^0-9]',
            r'cybersecurity\s+incident',
            r'material\s+cybersecurity'
        ]
        
        has_item_105 = any(re.search(pattern, content_lower) for pattern in item_105_patterns)
        
        if not has_item_105:
            return None
        
        # Score cybersecurity keywords
        found_keywords = []
        confidence_score = 0.0
        
        for keyword in self.cyber_keywords['primary']:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
                confidence_score += 0.3
        
        for keyword in self.cyber_keywords['secondary']:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
                confidence_score += 0.2
        
        for keyword in self.cyber_keywords['technical']:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
                confidence_score += 0.1
        
        if confidence_score < 0.2 or not found_keywords:
            return None
        
        # Extract incident description
        incident_description = self._extract_incident_section(content)
        incident_date = self._extract_incident_date(incident_description)
        
        return CyberIncident(
            company_name=filing_info.get('company_name', 'Unknown'),
            ticker=filing_info.get('ticker', ''),
            cik=filing_info.get('cik', ''),
            filing_date=filing_info.get('published', '')[:10],
            filing_url=filing_info.get('filing_url', ''),
            incident_description=incident_description[:1000],
            incident_date=incident_date,
            keywords=list(set(found_keywords)),
            raw_content_preview=content[:500],
            filing_type=filing_info.get('form_type', '8-K'),
            confidence_score=min(confidence_score, 1.0)
        )
    
    def _extract_incident_section(self, content: str) -> str:
        """Extract incident section from filing"""
        patterns = [
            r'item\s+1\.0?5.*?(?=item\s+[2-9]|signature|$)',
            r'cybersecurity\s+incident.*?(?=item|signature|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                section = match.group(0)
                return re.sub(r'\s+', ' ', section).strip()[:2000]
        
        return "Description not extracted"
    
    def _extract_incident_date(self, description: str) -> Optional[str]:
        """Extract incident date from description"""
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def monitor_cybersecurity_incidents_bulk(self, days_back: int = 14, max_filings: int = 1000) -> List[CyberIncident]:
        """Main monitoring function using bulk data"""
        logger.info(f"=== Starting BULK cybersecurity monitoring ===")
        logger.info(f"Monitoring period: {days_back} days")
        logger.info(f"Maximum filings: {max_filings}")
        
        self.api_responses = []
        
        # Get bulk 8-K filings
        logger.info("Fetching bulk 8-K filings from SEC daily indices...")
        filings = self.get_bulk_8k_filings(days_back, max_filings)
        
        if not filings:
            logger.error("❌ No filings retrieved from SEC bulk data")
            return []
        
        logger.info(f"✅ Retrieved {len(filings)} filings for analysis")
        
        incidents = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analyze filings with threading for speed
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_filing = {
                executor.submit(self.analyze_single_filing, filing): filing 
                for filing in filings
            }
            
            for i, future in enumerate(as_completed(future_to_filing)):
                progress = (i + 1) / len(filings)
                progress_bar.progress(progress)
                
                filing = future_to_filing[future]
                status_text.text(f'Analyzing {filing["company_name"]} ({i+1}/{len(filings)})')
                
                try:
                    incident = future.result()
                    if incident:
                        incidents.append(incident)
                        logger.info(f"🚨 CYBERSECURITY INCIDENT: {incident.company_name}")
                except Exception as e:
                    logger.error(f"Error analyzing {filing['company_name']}: {e}")
        
        progress_bar.progress(1.0)
        status_text.text(f'✅ Analyzed {len(filings)} filings, found {len(incidents)} incidents')
        
        logger.info(f"=== Bulk Monitoring Complete ===")
        logger.info(f"Total filings analyzed: {len(filings)}")
        logger.info(f"Cybersecurity incidents found: {len(incidents)}")
        
        incidents.sort(key=lambda x: x.confidence_score, reverse=True)
        return incidents
    
    def analyze_single_filing(self, filing: Dict) -> Optional[CyberIncident]:
        """Analyze a single filing for cybersecurity incidents"""
        content = self.analyze_8k_content(filing['filing_url'])
        if content:
            return self.detect_cybersecurity_incident(content, filing)
        return None
    
    def get_api_communication_log(self) -> pd.DataFrame:
        """Return detailed log of all API communications"""
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
                'Error': response.error or 'Success'
            })
        
        return pd.DataFrame(log_data)

def main():
    st.title("🔍 SEC 8-K Monitor - BULK DATA VERSION")
    st.markdown(f"**Version {VERSION} - Get 1000+ filings using SEC's bulk daily index files**")
    
    # Version badge
    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(90deg, #FF6B6B, #4ECDC4); padding: 8px; border-radius: 5px; margin-bottom: 10px;'>
        <strong style='color: white;'>🚀 Version {VERSION}</strong><br/>
        <small style='color: white; opacity: 0.9;'>Build: {BUILD_DATE}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    debug_mode = st.sidebar.checkbox("🐛 Enable Debug Mode", value=False)
    
    company_name = st.sidebar.text_input("Company Name", value="CyberSecurityTracker")
    email = st.sidebar.text_input("Contact Email", value="contact@company.com")
    
    if not company_name or not email or '@' not in email:
        st.error("⚠️ SEC requires valid company name and email")
        return
    
    days_back = st.sidebar.slider("Monitoring Period (days)", 1, 30, 14)
    
    max_filings = st.sidebar.selectbox(
        "Maximum Filings to Analyze",
        [100, 500, 1000, 2000, 5000],
        index=2,
        help="Higher = more comprehensive but slower"
    )
    
    min_confidence = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.2, 0.1)
    
    # Feature highlights
    st.sidebar.markdown("---")
    st.sidebar.markdown("**✨ Key Features:**")
    for feature in FEATURES[:3]:  # Show top 3 features
        st.sidebar.markdown(f"• {feature}")
    
    st.header("📊 BULK Data Collection")
    
    # Information about bulk data approach
    with st.expander("ℹ️ How Bulk Data Collection Works"):
        st.markdown("""
        **This version uses SEC's official bulk data system:**
        
        1. **Daily Index Files**: Downloads SEC's daily filing index files
        2. **Historical Coverage**: Scans back through multiple days of filings  
        3. **Company Mapping**: Maps CIKs to ticker symbols
        4. **Bulk Processing**: Analyzes hundreds/thousands of filings
        5. **Multi-threading**: Parallel processing for speed
        
        **Expected Results:**
        - 14 days ≈ 1,000-2,000 filings
        - 30 days ≈ 2,000-4,000 filings
        
        **Why this works better than RSS:**
        - RSS feeds limited to ~100 recent filings
        - Bulk data covers complete historical periods
        - Real control over volume and timeframe
        """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Ready to scan SEC's bulk data archives for cybersecurity incidents?**")
    with col2:
        run_monitor = st.button("🚀 Start BULK Scan", type="primary")
    
    if run_monitor:
        # Clear previous logs
        if hasattr(streamlit_handler, 'clear_logs'):
            streamlit_handler.clear_logs()
        
        # Initialize monitor
        with st.spinner("Initializing bulk data monitor..."):
            monitor = SECBulkDataMonitor(company_name, email, debug_mode)
        
        if debug_mode:
            with st.expander("🐛 System Debug Information", expanded=False):
                debug_info = {
                    'version': VERSION,
                    'build_date': BUILD_DATE,
                    'features': FEATURES,
                    'configuration': {
                        'user_agent': monitor.headers['User-Agent'],
                        'days_back': days_back,
                        'max_filings': max_filings,
                        'min_confidence': min_confidence,
                        'debug_mode': debug_mode
                    },
                    'endpoints': {
                        'bulk_data_base': monitor.bulk_data_base,
                        'company_tickers': monitor.company_tickers_url,
                        'submissions_api': monitor.submissions_api
                    },
                    'processing': {
                        'request_delay': monitor.request_delay,
                        'max_retries': monitor.max_retries,
                        'max_workers': 5
                    },
                    'keyword_categories': {
                        'primary': len(monitor.cyber_keywords['primary']),
                        'secondary': len(monitor.cyber_keywords['secondary']),
                        'technical': len(monitor.cyber_keywords['technical']),
                        'total_keywords': sum(len(v) for v in monitor.cyber_keywords.values())
                    }
                }
                st.json(debug_info)
        
        # Run the bulk scan
        with st.spinner(f"Scanning SEC bulk data for {days_back} days, up to {max_filings} filings..."):
            incidents = monitor.monitor_cybersecurity_incidents_bulk(days_back, max_filings)
        
        # Show debug information if enabled
        if debug_mode:
            st.header("🔗 SEC API Communication Log")
            api_log = monitor.get_api_communication_log()
            if not api_log.empty:
                st.dataframe(api_log, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total API Calls", len(api_log))
                with col2:
                    success_rate = len(api_log[api_log['Status Code'] == 200]) / len(api_log) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with col3:
                    total_data = api_log['Content Length'].sum()
                    st.metric("Total Data", f"{total_data:,} bytes")
                with col4:
                    avg_time = api_log['Response Time (s)'].mean()
                    st.metric("Avg Response Time", f"{avg_time:.2f}s")
            else:
                st.info("No API calls recorded")
            
            st.header("📋 System Logs")
            logs = streamlit_handler.get_logs()
            if logs:
                st.text_area("Debug Logs", logs, height=300)
            else:
                st.info("No system logs generated")
        
        # Filter incidents by confidence
        high_confidence_incidents = [i for i in incidents if i.confidence_score >= min_confidence]
        total_incidents = len(incidents)
        
        # Display results summary
        if incidents:
            st.success(f"🎉 BULK SCAN COMPLETE! Found {len(high_confidence_incidents)} high-confidence incidents out of {total_incidents} total detections.")
        else:
            st.warning(f"Bulk scan complete! No cybersecurity incidents detected in the analyzed filings over {days_back} days.")
            
        # Show detailed results if incidents found
        if high_confidence_incidents:
            st.header("🚨 Cybersecurity Incidents Detected")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("High Confidence", len(high_confidence_incidents))
            with col2:
                st.metric("Total Detections", total_incidents)
            with col3:
                if high_confidence_incidents:
                    avg_confidence = sum(i.confidence_score for i in high_confidence_incidents) / len(high_confidence_incidents)
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                else:
                    st.metric("Avg Confidence", "0.00")
            with col4:
                unique_companies = len(set(i.company_name for i in high_confidence_incidents))
                st.metric("Unique Companies", unique_companies)
            
            # Incidents table
            incidents_data = []
            for incident in high_confidence_incidents:
                incidents_data.append({
                    "Company": incident.company_name,
                    "Ticker": incident.ticker if incident.ticker else "N/A",
                    "Filing Date": incident.filing_date,
                    "Confidence": f"{incident.confidence_score:.2f}",
                    "Keywords Found": len(incident.keywords),
                    "Top Keywords": ", ".join(incident.keywords[:3]) + ("..." if len(incident.keywords) > 3 else "")
                })
            
            if incidents_data:
                df = pd.DataFrame(incidents_data)
                st.dataframe(df, use_container_width=True)
            
            # Detailed incident analysis
            st.subheader("📋 Detailed Incident Analysis")
            for i, incident in enumerate(high_confidence_incidents):
                with st.expander(f"#{i+1}: {incident.company_name} ({incident.ticker or 'No ticker'}) - Confidence: {incident.confidence_score:.2f}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**📝 Incident Description:**")
                        st.write(incident.incident_description)
                        
                        st.markdown("**🔗 SEC Filing:**")
                        st.link_button("View Original 8-K Filing", incident.filing_url)
                        
                        if debug_mode and incident.raw_content_preview:
                            st.markdown("**🐛 Raw Content Preview:**")
                            st.text_area(f"Raw content preview #{i+1}", incident.raw_content_preview, height=100)
                    
                    with col2:
                        st.markdown("**📊 Incident Metadata:**")
                        incident_metadata = {
                            "Company": incident.company_name,
                            "Ticker": incident.ticker or "Unknown",
                            "CIK": incident.cik,
                            "Filing Date": incident.filing_date,
                            "Incident Date": incident.incident_date or "Not specified",
                            "Filing Type": incident.filing_type,
                            "Confidence Score": round(incident.confidence_score, 3),
                            "Keywords Found": incident.keywords,
                            "Keyword Count": len(incident.keywords)
                        }
                        st.json(incident_metadata)
        
        elif total_incidents > 0:
            st.info(f"Found {total_incidents} potential incidents, but none met the confidence threshold of {min_confidence:.1f}")
            st.markdown("**💡 Try lowering the confidence threshold in the sidebar to see more results.**")
            
            if debug_mode:
                st.subheader("🔍 Low-Confidence Detections (Debug Only)")
                low_confidence_incidents = [i for i in incidents if i.confidence_score < min_confidence]
                for i, incident in enumerate(low_confidence_incidents[:5]):  # Show max 5
                    with st.expander(f"Low-conf #{i+1}: {incident.company_name} - Confidence: {incident.confidence_score:.2f}"):
                        st.write(f"**Keywords:** {', '.join(incident.keywords)}")
                        st.write(f"**Description Preview:** {incident.incident_description[:200]}...")
                        st.write(f"**Filing URL:** {incident.filing_url}")
        
        else:
            st.info(f"No cybersecurity incidents detected in the analyzed filings.")
            if debug_mode:
                st.markdown("**🔧 Troubleshooting:**")
                st.markdown("- Check the API Communication Log above for failed requests")  
                st.markdown("- Verify the date range covers recent activity")
                st.markdown("- Consider that cybersecurity incidents are relatively rare")
                st.markdown("- Try increasing the monitoring period or lowering confidence threshold")
    
    # Footer with version info
    st.markdown("---")
    st.markdown(f"**SEC 8-K Cybersecurity Monitor v{VERSION}** | Built: {BUILD_DATE} | Bulk Data Capability")

if __name__ == "__main__":
    main()
