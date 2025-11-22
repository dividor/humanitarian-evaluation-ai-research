#!/usr/bin/env python3
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

chrome_options = Options()
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.set_capability("goog:loggingPrefs", {"browser": "ALL"})

driver = webdriver.Chrome(options=chrome_options)

try:
    driver.get("http://localhost:3000")
    input(
        "Press Enter after you've searched for 'La revue documentaire' and clicked first result..."
    )

    # Get browser console logs
    logs = driver.get_log("browser")
    print("\n📝 Browser Console Logs:")
    for log in logs:
        print(f"{log['level']}: {log['message']}")

    # Check current page state
    try:
        page_input = driver.find_element(By.CSS_SELECTOR, "input[type='number']")
        current_page = page_input.get_attribute("value")
        print(f"\nCurrent page: {current_page}")
    except Exception:
        print("\nCould not find page input")

    # Check scroll position
    try:
        scroll_container = driver.find_element(By.CSS_SELECTOR, ".pdf-viewer-content")
        scroll_top = driver.execute_script(
            "return arguments[0].scrollTop;", scroll_container
        )
        scroll_height = driver.execute_script(
            "return arguments[0].scrollHeight;", scroll_container
        )
        print(f"Scroll position: {scroll_top}px / {scroll_height}px")
    except Exception as e:
        print(f"Could not get scroll info: {e}")

    # Check for canvas
    canvases = driver.find_elements(By.CSS_SELECTOR, ".pdf-viewer-content canvas")
    print(f"Canvases found: {len(canvases)}")

    # Check for highlights
    highlights = driver.find_elements(By.CSS_SELECTOR, ".highlight-overlay")
    print(f"Highlights found: {len(highlights)}")

    driver.save_screenshot("manual_test.png")
    print("\n📸 Screenshot saved: manual_test.png")

    input("Press Enter to close browser...")

finally:
    driver.quit()
