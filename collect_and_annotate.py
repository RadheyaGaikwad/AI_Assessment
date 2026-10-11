"""
Phase 1: Automated Collection & Zero-Shot Annotation
File: collect_and_annotate.py
Description: Systematic web scraping using Playwright to collect high-resolution 
human pose images from verified category sources, ensuring clean directory structure 
and duplicate elimination.
"""

import os
import time
import requests
from urllib.parse import urlparse, urlunparse
from playwright.sync_api import sync_playwright

def scrape_human_pose_images():
    """
    Data Engineering: Executes headless browser automation via Playwright 
    to scrape target image URLs, filtering out SVGs and duplicates, and 
    storing raw image assets safely without overwriting existing files.
    """
    output_dir = "data/raw_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check existing files to maintain incremental collection and state preservation
    existing_files = os.listdir(output_dir)
    start_index = len([f for f in existing_files if f.endswith('.jpg') or f.endswith('.png')])
    print(f"[-] Found {start_index} existing images. Preserving them safely.")

    target_per_category = 35
    image_urls = set()

    # Verified high-reliability Wikimedia category endpoints for human pose variations[cite: 14]
    categories = [
        "https://commons.wikimedia.org/wiki/Category:People_standing",
        "https://commons.wikimedia.org/wiki/Category:Sitting",
        "https://commons.wikimedia.org/wiki/Category:Kneeling",
        "https://commons.wikimedia.org/wiki/Category:Human_postures"
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for search_url in categories:
            print(f"\n[-] Navigating to category: {search_url}")
            try:
                page.goto(search_url, timeout=30000)
                page.wait_for_load_state("domcontentloaded")
                time.sleep(2)
            except Exception as e:
                print(f"[!] Skipping category due to navigation error: {e}")
                continue

            scroll_attempts = 0
            max_scrolls = 4
            collected_for_this_cat = 0

            while scroll_attempts < max_scrolls and collected_for_this_cat < target_per_category:
                try:
                    links = page.evaluate("""() => {
                        const imgs = document.querySelectorAll('.gallerybox img, .filethumb img, ul.gallery img, .mw-file-description img');
                        return Array.from(imgs).map(img => img.src || img.getAttribute('data-src'));
                    }""")
                    
                    before_count = len(image_urls)
                    for src in links:
                        if src:
                            if src.startswith("//"):
                                src = "https:" + src
                            
                            # Normalize thumbnail URLs to fetch full-resolution assets
                            if "/thumb/" in src:
                                parts = src.split("/thumb/")
                                base_url = parts[0]
                                sub_path = parts[1].rsplit('/', 1)[0]
                                src = base_url + "/" + sub_path
                            
                            parsed_url = urlparse(src)
                            clean_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))
                                
                            if any(ext in clean_url.lower() for ext in ['.jpg', '.jpeg', '.png']) and 'svg' not in clean_url.lower():
                                if clean_url not in image_urls:
                                    image_urls.add(clean_url)
                                    collected_for_this_cat += 1

                    print(f"[+] Total unique URLs collected so far: {len(image_urls)}")
                    page.mouse.wheel(0, 1500)
                    time.sleep(1.5)
                    
                    if len(image_urls) == before_count:
                        break
                    scroll_attempts += 1
                except Exception as loop_err:
                    print(f"[!] Loop note: {loop_err}")
                    break

        urls_to_download = list(image_urls)
        print(f"\n[-] Downloading clean images to {output_dir}...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://commons.wikimedia.org/"
        }

        downloaded = 0
        for i, url in enumerate(urls_to_download):
            file_index = start_index + downloaded + 1
            file_path = os.path.join(output_dir, f"pose_{file_index:03d}.jpg")
            
            if os.path.exists(file_path):
                continue
                
            success = False
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    downloaded += 1
                    success = True
            except Exception:
                pass

            if not success:
                try:
                    img_bytes = page.request.get(url).body()
                    if img_bytes:
                        with open(file_path, "wb") as f:
                            f.write(img_bytes)
                        downloaded += 1
                except Exception as ex:
                    print(f"[WARN] Skipped {url} due to download error: {ex}")

        total_images = len([f for f in os.listdir(output_dir) if f.endswith('.jpg') or f.endswith('.png')])
        print(f"\n[✔] Done! Total dataset size now: {total_images} images in {output_dir}.")
        try:
            browser.close()
        except Exception:
            pass

if __name__ == "__main__":
    scrape_human_pose_images()