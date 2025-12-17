"""
Button clicking strategies for Selenium crawler.
Handles different methods to find and click "Load More" buttons.
"""
import time
import logging
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class ButtonClickStrategy:
    """Strategies to find and click load more buttons."""
    
    def __init__(self, driver):
        self.driver = driver
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def find_button(self, selector):
        """Try multiple methods to find the button."""
        strategies = [
            self._find_by_css_selector(selector),
            self._find_by_text_content(),
            self._find_by_id_pattern(),
            self._find_by_class_pattern()
        ]
        
        for strategy in strategies:
            button = strategy
            if button:
                return button
        
        return None
    
    def _find_by_css_selector(self, selector):
        """Method 1: Find by CSS selector."""
        try:
            button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            self.logger.info(f"Found button with selector: {selector}")
            return button
        except:
            return None
    
    def _find_by_text_content(self):
        """Method 2: Find by text 'আরও'."""
        try:
            button = self.driver.find_element(
                By.XPATH, 
                "//button[contains(., 'আরও')] | //a[contains(., 'আরও')] | //*[contains(@class, 'load') and contains(., 'আরও')]"
            )
            self.logger.info("Found button by text 'আরও'")
            return button
        except:
            return None
    
    def _find_by_id_pattern(self):
        """Method 3: Find by ID pattern."""
        try:
            button = self.driver.find_element(
                By.CSS_SELECTOR, 
                "button[id*='ajax_load_more'], a[id*='ajax_load_more'], button[id*='load_more'], a[id*='load_more']"
            )
            self.logger.info("Found button by ID pattern")
            return button
        except:
            return None
    
    def _find_by_class_pattern(self):
        """Method 4: Find by class pattern."""
        try:
            button = self.driver.find_element(
                By.CSS_SELECTOR, 
                "button[class*='load-more'], a[class*='load-more'], button[class*='more'], a[class*='more']"
            )
            self.logger.info("Found button by class pattern")
            return button
        except:
            return None
    
    def click_button(self, button):
        """Try to click the button using multiple methods."""
        # Scroll to button
        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
        time.sleep(1.0)
        
        # Try regular click if visible
        try:
            if button.is_displayed():
                button.click()
                self.logger.info("Clicked button with regular click")
                return True
        except:
            pass
        
        # Fallback to JavaScript click
        try:
            self.driver.execute_script("arguments[0].click();", button)
            self.logger.info("Clicked button with JavaScript")
            return True
        except Exception as e:
            self.logger.error(f"Failed to click button: {e}")
            return False
    
    def handle_infinite_scroll(self, wait_time=20):
        """Handle infinite scroll loading."""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(wait_time)
        
        new_height = self.driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            self.logger.info("No more content loading with infinite scroll")
            return False
        else:
            self.logger.info("Infinite scroll loaded more content")
            return True
