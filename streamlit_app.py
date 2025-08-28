import streamlit as st
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class ProductionEdgarMonitor:
    """Production SEC EDGAR monitor with real API calls"""
    
    def __init__(self, company_name: str = "CyberIncidentTracker", email: str = "contact@company.com"):
        # SEC requires proper User-Agent with company name and email
        self.headers = {
            "User-Agent": f"{company_name} {email}",
            "Accept": "application/json, text/html, application/xml",
            "Accept-Encoding": "gzip, deflate"
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # SEC EDGAR endpoints
        self.edgar_base = "https://www.sec.gov"
        self.edgar_search_base = "https://efts.sec.gov/LATEST/search-index"
        self.edgar_rss = "https://www.sec.gov/cgi-bin/browse-edgar"
        
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
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Make rate-limited request to SEC with retries"""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"All requests failed for {url}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def get_recent_8k_filings_rss(self, days_back: int = 7) -> List[Dict]:
        """Get recent 8-K filings using SEC RSS feed"""
        logger.info(f"Fetching 8-K filings from last {days_back} days via RSS")
        
        # SEC RSS parameters for 8-K filings
        params = {
            'action': 'getcurrent',
            'type': '8-K',
            'output': 'atom',
            'count': min(100, days_back * 20)  # Estimate filings per day
        }
        
        response = self._make_request(self.edgar_rss, params)
        if not response:
            return []
        
        try:
            # Parse RSS/Atom feed
            feed = feedparser.parse(response.content)
            filings = []
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for entry in feed.entries:
                # Parse entry date
                try:
                    entry_date = datetime(*entry.published_parsed[:6])
                    if entry_date < cutoff_date:
                        continue
                except:
                    continue
                
                # Extract company info from title
                title = entry.title
                company_match = re.search(r'8-K\s*-\s*(.+?)\s*\(([^)]+)\)', title)
                if company_match:
                    company_name = company_match.group(1).strip()
                    cik_match = re.search(r'\b(\d{10})\b', company_match.group(2))
                    cik = cik_match.group(1) if cik_match else ""
                    
                    filing = {
                        'title': title,
                        'company_name': company_name,
                        'cik': cik,
                        'filing_url': entry.link,
                        'published': entry_date.isoformat(),
                        'summary': getattr(entry, 'summary', '')
                    }
                    filings.append(filing)
            
            logger.info(f"Found {len(filings)} recent 8-K filings")
            return filings
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed: {e}")
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
        """Download and extract text content from 8-K filing"""
        logger.info(f"Analyzing content: {filing_url}")
        
        response = self._make_request(filing_url)
        if not response:
            return None
        
        try:
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error parsing filing content: {e}")
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
    
    def monitor_cybersecurity_incidents(self, days_back: int = 7, use_rss: bool = True) -> List[CyberIncident]:
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
    """Production Streamlit app with real SEC data"""
    
    st.title("🔍 SEC 8-K Cybersecurity Monitor - Production")
    st.markdown("**Real-time monitoring of cybersecurity incidents using live SEC EDGAR data**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
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
        value=0.5,
        step=0.1,
        help="Filter incidents by detection confidence"
    )
    
    # Main monitoring
    st.header("📊 Live Incident Detection")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Scan live SEC EDGAR filings for cybersecurity incidents")
    with col2:
        run_monitor = st.button("🔍 Start Scan", type="primary")
    
    if run_monitor:
        # Initialize production monitor
        with st.spinner("Initializing SEC EDGAR connection..."):
            monitor = ProductionEdgarMonitor(company_name, email)
        
        # Run monitoring
        with st.spinner("Scanning SEC filings..."):
            incidents = monitor.monitor_cybersecurity_incidents(days_back, use_rss)
        
        # Filter by confidence
        high_confidence_incidents = [i for i in incidents if i.confidence_score >= min_confidence]
        
        st.success(f"Scan complete! Found {len(high_confidence_incidents)} high-confidence incidents out of {len(incidents)} total detections.")
        
        if high_confidence_incidents:
            # Display results
            st.header("🚨 Cybersecurity Incidents Detected")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("High Confidence", len(high_confidence_incidents))
            with col2:
                st.metric("Total Detections", len(incidents))
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
        else:
            st.info(f"No high-confidence cybersecurity incidents found in the last {days_back} days.")
    
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
    - ✅ Detailed logging and monitoring
    
    **Rate Limits:** Respects SEC's 10 requests/second limit
    **Data Sources:** Live SEC EDGAR RSS feeds and search API
    **Detection:** Multi-tier keyword analysis with confidence scoring
    """)

if __name__ == "__main__":
    main()
