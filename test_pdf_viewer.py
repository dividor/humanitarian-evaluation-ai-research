#!/usr/bin/env python3
"""
Test clicking on multiple search results to see which ones work
"""
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=chrome_options)

try:
    driver.get("http://localhost:3000")
    print("Navigated to app")

    search_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']"))
    )
    search_input.clear()
    search_input.send_keys("evaluation")

    print("Waiting for search results...")
    time.sleep(6)  # Wait for debounce + search

    results = driver.find_elements(By.CSS_SELECTOR, ".search-result")
    print(f"\nFound {len(results)} results\n")

    # Test first 5 results
    for i in range(min(5, len(results))):
        print(f"{'='*60}")
        print(f"Testing Result #{i+1}")
        print(f"{'='*60}")

        # Re-find results (DOM might have changed)
        results = driver.find_elements(By.CSS_SELECTOR, ".search-result")
        if i >= len(results):
            print(f"Result #{i+1} not found")
            continue

        result = results[i]

        # Get title
        try:
            title_elem = result.find_element(
                By.CSS_SELECTOR, ".result-title, h3, .search-result-title"
            )
            title = title_elem.text[:60]
            print(f"Title: {title}...")
        except Exception:
            title = "Unknown"
            print("Title: Unknown")

        # Click result
        driver.execute_script("arguments[0].scrollIntoView(true);", result)
        time.sleep(0.3)
        result.click()

        # Wait for PDF viewer
        time.sleep(2)

        # Check page number
        try:
            page_input = driver.find_element(By.CSS_SELECTOR, "input[type='number']")
            current_page = page_input.get_attribute("value")
            total_text = driver.find_element(
                By.XPATH, "//span[contains(text(), 'of')]"
            ).text
            print(f"Page indicator: {current_page} {total_text}")
        except Exception as e:
            print(f"Could not read page: {e}")
            current_page = "?"

        # Check if PDF content is visible (look for canvas)
        try:
            driver.find_element(By.CSS_SELECTOR, ".pdf-viewer-content canvas")
            print("✅ Canvas found")
        except Exception:
            print("❌ No canvas found - PDF not rendered")

        # Check for highlights
        highlights = driver.find_elements(By.CSS_SELECTOR, ".highlight-overlay")
        print(f"Highlights found: {len(highlights)}")

        # Take screenshot
        driver.save_screenshot(f"test_result_{i+1}.png")
        print(f"📸 Screenshot: test_result_{i+1}.png")

        # Close PDF viewer
        try:
            close_btn = driver.find_element(
                By.CSS_SELECTOR, ".close-button, button[aria-label='Close']"
            )
            close_btn.click()
            time.sleep(0.5)
        except Exception as e:
            print(f"Could not close PDF viewer: {e}")
            # Try ESC key
            from selenium.webdriver.common.keys import Keys

            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
            time.sleep(0.5)

        print()

    print("✅ Test completed")

finally:
    driver.quit()
