#!/usr/bin/env python3
"""
Test PDF highlighting for multiple search results
"""
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def test_pdf_highlighting():
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 20)

    try:
        print("\n=== Testing PDF Highlighting ===")

        # Navigate to the app
        print("\n1. Loading app...")
        driver.get("http://localhost:3000")
        time.sleep(2)

        # Search for the specific text
        search_text = (
            "L'élaboration  d'un  guide  sur  l'accès  à  la  sécurité  "
            "sociale  a  démarré  au  mois d'avril et livré en juin 2022. Il a"
        )
        print(f"\n2. Searching for: {search_text[:50]}...")
        search_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="text"]'))
        )
        search_input.send_keys(search_text)

        # Wait for results
        print("\n3. Waiting for search results...")
        time.sleep(5)

        # Find result cards
        result_cards = driver.find_elements(By.CSS_SELECTOR, ".result-card")
        if not result_cards:
            print("❌ No search results found!")
            return

        print(f"   Found {len(result_cards)} search results")

        # Test top 5 results
        num_to_test = min(5, len(result_cards))
        for i in range(num_to_test):
            print(f"\n{'='*60}")
            print(f"Testing Result #{i+1}")
            print(f"{'='*60}")

            # Click the result
            result_card = result_cards[i]
            title_link = result_card.find_element(By.CSS_SELECTOR, ".result-title-link")
            result_title = title_link.text
            print(f"   Title: {result_title[:60]}...")

            # Use JavaScript to click
            driver.execute_script("arguments[0].click();", title_link)
            time.sleep(1)

            # Wait for PDF viewer
            wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "pdf-viewer-container"))
            )
            print("   ✅ PDF viewer opened")

            # Wait for rendering
            time.sleep(3)

            # Check for highlights
            highlights = driver.find_elements(By.CSS_SELECTOR, ".highlight-overlay")
            print(f"   Highlights found: {len(highlights)}")

            if len(highlights) == 0:
                print("   ❌ No highlights rendered!")
            else:
                # Get first highlight details
                first_highlight = highlights[0]
                highlight_visible = first_highlight.is_displayed()
                print(f"   First highlight visible: {highlight_visible}")

                if highlight_visible:
                    # Get position
                    location = first_highlight.location
                    size = first_highlight.size
                    print(
                        f"   Highlight position: x={location['x']}, "
                        f"y={location['y']}, "
                        f"w={size['width']}, h={size['height']}"
                    )

                    # Check if centered in viewport
                    viewport_height = driver.execute_script(
                        "return window.innerHeight;"
                    )
                    highlight_middle = location["y"] + size["height"] / 2
                    viewport_middle = viewport_height / 2

                    distance_from_center = abs(highlight_middle - viewport_middle)
                    print(f"   Viewport height: {viewport_height}px")
                    print(f"   Highlight middle: {highlight_middle}px")
                    print(f"   Viewport middle: {viewport_middle}px")
                    print(f"   Distance from center: {distance_from_center}px")

                    # Consider "centered" if within 200px of center
                    is_centered = distance_from_center < 300
                    if is_centered:
                        print("   ✅ Highlight is reasonably centered")
                    else:
                        print(
                            f"   ⚠️  Highlight is {distance_from_center}px from center"
                        )

                # Wait a moment to see if highlight disappears
                print("   Checking if highlight persists...")
                time.sleep(2)
                highlights_after = driver.find_elements(
                    By.CSS_SELECTOR, ".highlight-overlay"
                )
                if len(highlights_after) == 0:
                    print("   ❌ HIGHLIGHT DISAPPEARED!")
                else:
                    print(f"   ✅ Highlight still visible ({len(highlights_after)})")

            # Take screenshot
            screenshot_path = (
                "/Users/matthewharris/Desktop/git/"
                "humanitarian-evaluation-ai-research/"
                f"highlight_test_result_{i+1}.png"
            )
            driver.save_screenshot(screenshot_path)
            print(f"   📸 Screenshot: highlight_test_result_{i+1}.png")

            # Close PDF viewer
            close_button = driver.find_element(By.CLASS_NAME, "close-button")
            close_button.click()
            time.sleep(1)
            print("   PDF viewer closed\n")

        print("\n✅ All tests completed")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()

        # Try to take screenshot on error
        try:
            error_path = (
                "/Users/matthewharris/Desktop/git/"
                "humanitarian-evaluation-ai-research/highlight_error.png"
            )
            driver.save_screenshot(error_path)
            print("Error screenshot saved")
        except Exception:
            pass

    finally:
        driver.quit()


if __name__ == "__main__":
    test_pdf_highlighting()
