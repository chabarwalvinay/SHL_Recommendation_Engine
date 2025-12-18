import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random  # Import random for human-like delays
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import json

BASE_URL = "https://www.shl.com"
BASE_CATALOG_URL = "https://www.shl.com/products/product-catalog/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


def get_session():
    session = requests.Session()
    session.headers.update(HEADERS)

    retry_strategy = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


crawler_session = get_session()


def get_soup(url: str) -> BeautifulSoup:
    resp = crawler_session.get(url, timeout=60)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_individual_test_links(page_soup: BeautifulSoup) -> list[str]:
    links = set()
    product_rows = page_soup.find_all("tr", attrs={"data-entity-id": True})

    for row in product_rows:
        title_col = row.find("td", class_="custom__table-heading__title")
        if title_col:
            a_tag = title_col.find("a", href=True)
            if a_tag:
                links.add(urljoin(BASE_URL, a_tag["href"]))
    return list(links)


def extract_test_details(url: str) -> dict | None:
    try:
        soup = get_soup(url)

        # --- 1. NAME ---
        name_tag = soup.find("h1")
        if not name_tag:
            return None
        name = name_tag.get_text(strip=True)

        # --- 2. DESCRIPTION ---
        description = ""
        desc_header = soup.find("h4", string=re.compile("Description", re.I))
        if desc_header:
            desc_content = desc_header.find_next("p")
            if desc_content:
                description = desc_content.get_text(" ", strip=True)

        if not description:
            rich_text = soup.find("div", class_="rich-text")
            if rich_text:
                description = rich_text.get_text(" ", strip=True)

        # --- 3. DURATION ---
        duration = 0
        length_header = soup.find("h4", string=re.compile("Assessment length", re.I))
        if length_header:
            length_text_tag = length_header.find_next("p")
            if length_text_tag:
                match = re.search(r"(\d+)", length_text_tag.get_text(strip=True))
                if match:
                    duration = int(match.group(1))

        # --- 4. REMOTE SUPPORT ---
        remote_support = "No"
        remote_label = soup.find(string=re.compile("Remote Testing", re.I))
        if remote_label:
            parent = remote_label.parent
            if parent:
                yes_circle = parent.find("span", class_=lambda x: x and "-yes" in x)
                if not yes_circle and parent.parent:
                    yes_circle = parent.parent.find(
                        "span", class_=lambda x: x and "-yes" in x
                    )
                if yes_circle:
                    remote_support = "Yes"

        # --- 5. TEST TYPE (FIXED & SCOPED) ---
        test_type_list = []

        type_label = soup.find(string=re.compile(r"Test Type\s*:", re.I))
        if type_label:
            container_p = type_label.find_parent("p")
            if container_p:
                key_spans = container_p.select("span.product-catalogue__key")
                test_type_list = [
                    span.get_text(strip=True)
                    for span in key_spans
                    if span.get_text(strip=True)
                ]

        test_type_list = list(set(test_type_list))

        return {
            "name": name,
            "url": url,
            "adaptive_support": "No",
            "description": description,
            "duration": duration,
            "remote_support": remote_support,
            "test_type": test_type_list,
        }

    except requests.exceptions.HTTPError as e:
        print(f"Skipping {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None


def crawl_shl_assessments() -> list[dict]:
    assessments = []
    seen_urls = set()

    offset = 0
    items_per_page = 12
    page_number = 1

    print("Starting robust ingestion...")

    while True:
        current_url = f"{BASE_CATALOG_URL}?start={offset}&type=1"
        print(f"Crawling Page {page_number} (offset={offset})...")

        try:
            soup = get_soup(current_url)
        except Exception as e:
            print(f"CRITICAL: Failed to load catalog page {page_number}: {e}")
            break

        new_links = extract_individual_test_links(soup)
        if not new_links:
            print(f"No products found at offset {offset}. End of catalog.")
            break

        count_new_on_page = 0
        for link in tqdm(new_links, leave=False, desc=f"Scraping Page {page_number}"):
            if link in seen_urls:
                continue

            seen_urls.add(link)
            data = extract_test_details(link)
            if data:
                assessments.append(data)
                count_new_on_page += 1

            time.sleep(random.uniform(2.5, 4.5))

        print(f" -> Added {count_new_on_page} items. (Total: {len(assessments)})")

        with open("assessments_data.json", "w") as f:
            json.dump(assessments, f, indent=4)

        offset += items_per_page
        page_number += 1
        time.sleep(2)

        if page_number > 60:
            break

    return assessments


if __name__ == "__main__":
    data = crawl_shl_assessments()
    print(f"Total assessments scraped: {len(data)}")

    if len(data) < 100:
        print("WARNING: Data count low. Check for blocks.")
    else:
        print("SUCCESS.")
