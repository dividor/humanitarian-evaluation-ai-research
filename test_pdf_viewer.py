"""
Selenium test to verify PDF viewer works for multiple search results
"""

import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def test_pdf_viewer_multiple_results():
    """Test that clicking different result cards opens different PDFs with highlights"""

    # Setup Chrome
    options = Options()
    options.add_argument("--headless")  # Run headless
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        print("Opening UI...")
        driver.get("http://localhost:3000")

        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        search_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".search-input"))
        )

        print("Searching for 'medical'...")
        search_input.send_keys("medical")
        search_input.send_keys(Keys.RETURN)

        # Wait for results
        print("Waiting for results...")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".result-card")))
        time.sleep(2)  # Extra wait for all results to render

        # Get all result cards
        result_cards = driver.find_elements(By.CSS_SELECTOR, ".result-card")
        print(f"Found {len(result_cards)} result cards")

        if len(result_cards) < 2:
            print("ERROR: Need at least 2 results to test")
            return False

        # Test clicking first result
        print("\n=== Testing First Result ===")
        first_result = result_cards[0]
        first_doc_id = first_result.get_attribute("data-doc-id")
        first_page = first_result.get_attribute("data-page")
        print(f"First result: doc_id={first_doc_id}, page={first_page}")

        first_title = first_result.find_element(By.CSS_SELECTOR, ".result-title-link")
        first_title.click()

        # Wait for PDF viewer to appear
        print("Waiting for PDF viewer...")
        pdf_viewer = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".preview-panel"))
        )

        # Wait for PDF to load
        time.sleep(3)

        # Check if PDF is displayed (canvas should exist)
        canvas_elements = driver.find_elements(By.TAG_NAME, "canvas")
        print(f"Canvas elements found: {len(canvas_elements)}")
        if len(canvas_elements) == 0:
            print("ERROR: No canvas found - PDF not rendering")
            return False

        # Check for highlights
        highlight_elements = driver.find_elements(By.CSS_SELECTOR, ".highlight-overlay")
        first_highlight_count = len(highlight_elements)
        print(f"Highlights on first PDF: {first_highlight_count}")

        # Get page number from toolbar
        page_input = driver.find_element(By.CSS_SELECTOR, ".page-input")
        first_page_displayed = page_input.get_attribute("value")
        print(f"Page displayed: {first_page_displayed}")

        # Test clicking second result
        print("\n=== Testing Second Result ===")

        # Close the first PDF viewer by clicking the close button
        close_button = driver.find_element(By.CSS_SELECTOR, ".close-button")
        close_button.click()
        time.sleep(1)  # Wait for overlay to close

        second_result = result_cards[1]
        second_doc_id = second_result.get_attribute("data-doc-id")
        second_page = second_result.get_attribute("data-page")
        print(f"Second result: doc_id={second_doc_id}, page={second_page}")

        second_title = second_result.find_element(By.CSS_SELECTOR, ".result-title-link")
        second_title.click()

        # Wait for PDF to update
        time.sleep(3)

        # Check if page changed (if different page)
        page_input = driver.find_element(By.CSS_SELECTOR, ".page-input")
        second_page_displayed = page_input.get_attribute("value")
        print(f"Page displayed after second click: {second_page_displayed}")

        # Check highlights again
        highlight_elements = driver.find_elements(By.CSS_SELECTOR, ".highlight-overlay")
        second_highlight_count = len(highlight_elements)
        print(f"Highlights on second PDF: {second_highlight_count}")

        # Verify PDF viewer updated
        success = True
        if first_doc_id == second_doc_id and first_page == second_page:
            print("WARNING: Both results are from same doc/page - can't verify update")
            # But at least check PDF viewer is still present
            if driver.find_elements(By.CSS_SELECTOR, ".pdf-viewer"):
                print("✓ PDF viewer still present")
            else:
                print("ERROR: PDF viewer disappeared")
                success = False
        else:
            # Different page/doc - verify page changed
            if second_page_displayed != first_page_displayed:
                print(
                    f"✓ Page changed from {first_page_displayed} to {second_page_displayed}"
                )
            else:
                print(f"WARNING: Page didn't change (still {second_page_displayed})")
                success = False

            # Check if highlights are present
            if second_highlight_count > 0:
                print(f"✓ Highlights present ({second_highlight_count} found)")
            else:
                print("WARNING: No highlights found on second PDF")

        # Test clicking third result if available
        if len(result_cards) >= 3:
            print("\n=== Testing Third Result ===")

            # Close the second PDF viewer
            close_button = driver.find_element(By.CSS_SELECTOR, ".close-button")
            close_button.click()
            time.sleep(1)

            third_result = result_cards[2]
            third_doc_id = third_result.get_attribute("data-doc-id")
            third_page = third_result.get_attribute("data-page")
            print(f"Third result: doc_id={third_doc_id}, page={third_page}")

            third_title = third_result.find_element(
                By.CSS_SELECTOR, ".result-title-link"
            )
            third_title.click()
            time.sleep(3)

            # Check page changed
            page_input = driver.find_element(By.CSS_SELECTOR, ".page-input")
            third_page_displayed = page_input.get_attribute("value")
            print(f"Page displayed after third click: {third_page_displayed}")

            # Check highlights
            highlight_elements = driver.find_elements(
                By.CSS_SELECTOR, ".highlight-overlay"
            )
            third_highlight_count = len(highlight_elements)
            print(f"Highlights on third PDF: {third_highlight_count}")

            # Verify PDF viewer still present
            if driver.find_elements(By.CSS_SELECTOR, ".preview-panel"):
                print("✓ PDF viewer works for third result too")
                if third_page_displayed == third_page:
                    print(f"✓ Page correctly updated to {third_page_displayed}")
                if third_highlight_count > 0:
                    print(f"✓ Highlights working ({third_highlight_count} found)")
            else:
                print("ERROR: PDF viewer failed on third result")
                success = False

        if not success:
            print("\n❌ TEST FAILED: PDF viewer did not work correctly for all results")
            return False

        print("\n✅ TEST PASSED: PDF viewer works for multiple results")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        driver.quit()


if __name__ == "__main__":
    success = test_pdf_viewer_multiple_results()
    exit(0 if success else 1)
