"""
Selenium WebDriver management and setup.
Handles driver initialization, configuration, and cleanup.
"""
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


class SeleniumDriverManager:
    """Manages Selenium WebDriver lifecycle and basic operations."""
    
    def __init__(self, headless=True):
        self.driver = None
        self.headless = headless
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def setup(self):
        """Initialize Chrome driver with optimal settings."""
        if self.driver:
            return
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # Anti-detection
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Hide webdriver property
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
        
        self.logger.info("Selenium driver initialized")
    
    def teardown(self):
        """Close and cleanup driver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.logger.info("Selenium driver closed")
    
    def get_page(self, url, wait_time=10, delay=2):
        """Load a page and wait for content."""
        self.setup()
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(delay)
            return BeautifulSoup(self.driver.page_source, 'lxml')
        except Exception as e:
            self.logger.error(f"Error loading {url}: {e}")
            return None
    
    def get_current_soup(self):
        """Get BeautifulSoup of current page."""
        if not self.driver:
            return None
        return BeautifulSoup(self.driver.page_source, 'lxml')
    
    def scroll_to_bottom(self):
        """Scroll page to bottom."""
        if self.driver:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    def get_page_height(self):
        """Get current page height."""
        if self.driver:
            return self.driver.execute_script("return document.body.scrollHeight")
        return 0
