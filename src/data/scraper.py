# Library Imports 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import time
from tqdm.auto import tqdm

def make_driver():
    """Create and configure Chrome WebDriver with headless options"""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--window-size=2050,2050")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_links_for_all_countries(url, driver):
    """Finds the container for country links and extracts the href from each <a> tag."""
    all_countries_links = []
    driver.get(url)
    time.sleep(3) 

    try:
        links_container = driver.find_element(By.ID, "myUL")
        anchor_tags = links_container.find_elements(By.TAG_NAME, "a")
        
        print(f"Found {len(anchor_tags)} links. Extracting URLs...")

        for tag in anchor_tags:
            href = tag.get_attribute('href')
            if href:
                all_countries_links.append(href)
                
    except NoSuchElementException:
        print("Could not find the container for the country links.")
        
    return all_countries_links

def scrape_country_data(country_url, driver):
    """
    Navigates to a country page and scrapes tables by detecting their
    specific column structure (3-column or multi-column).
    """
    driver.get(country_url)
    time.sleep(2)
    scraped_data = {}
    try:
        country_name = driver.find_element(By.CSS_SELECTOR, "td.countrytable").text
        print(f"\n--- Scraping data for: {country_name} ---")
        scraped_data['country_name'] = country_name

        section_headers = driver.find_elements(By.TAG_NAME, "summary")
        for header in section_headers:
            section_title = header.text
            table_data = {}
            try:
                driver.execute_script("arguments[0].click();", header)
                time.sleep(0.5)
            except Exception as e:
                print(f"Could not click header '{section_title}': {e}")
                continue

            parent_details = header.find_element(By.XPATH, "..")
            rows = parent_details.find_elements(By.CSS_SELECTOR, "tbody > tr")
            if not rows:
                continue
            
            num_columns = len(rows[0].find_elements(By.TAG_NAME, "td"))
            if num_columns == 3:
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) == 3:
                        key = cells[0].text.strip()
                        value = cells[2].text.strip()
                        if key:
                            table_data[key] = value
            elif num_columns > 3:
                try:
                    header_row = parent_details.find_element(By.CSS_SELECTOR, "thead > tr")
                    header_cells = [cell.text.strip() for cell in header_row.find_elements(By.TAG_NAME, "td")]
                except NoSuchElementException:
                     header_cells = [cell.text.strip() for cell in rows[0].find_elements(By.TAG_NAME, "td")]
                     rows = rows[1:]
                
                year_headers = header_cells[1:]
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) == num_columns:
                        indicator_name = cells[0].text.strip()
                        values = [cell.text.strip() for cell in cells[1:]]
                        if indicator_name:
                            table_data[indicator_name] = dict(zip(year_headers, values))
            scraped_data[section_title] = table_data
            
    except Exception as e:
        print(f"An unexpected error occurred on {country_url}: {e}")
    
    return scraped_data

def process_data_to_df(scraped_data_list):
    """
    Processes the raw list of nested dictionaries from the scraper
    into a clean, flat Pandas DataFrame.
    """
    records = []
    for country_data in scraped_data_list:
        country_name = country_data.get('country_name', 'Unknown')
        
        for category, indicators in country_data.items():
            if category == 'country_name':
                continue # Skip the country_name key itself
            
            for indicator, value in indicators.items():
                # Case 1: The value is a dictionary with years (e.g., Economic indicators)
                if isinstance(value, dict):
                    for year, val in value.items():
                        records.append({
                            'Country': country_name,
                            'Category': category,
                            'Indicator': indicator,
                            'Year': year,
                            'Value': val
                        })
                # Case 2: The value is a simple string (e.g., General Information)
                else:
                    records.append({
                        'Country': country_name,
                        'Category': category,
                        'Indicator': indicator,
                        'Year': 'N/A', # No year applicable for this type of data
                        'Value': value
                    })
                    
    return pd.DataFrame(records)


if __name__ == "__main__":
    #Link to the webpage
    link = "https://data.un.org/en/index.html"
    driver = make_driver()
    countries_links = get_links_for_all_countries(link, driver)
    print("Step 1: Collecting all country links...")
    print(f"Extracted all countries URL: {len(countries_links)}")

    all_data = []
    print(f"\nStep 2: Scraping data from all country links")
    for country_link in tqdm(countries_links, desc="Scraping Countries"):
        data = scrape_country_data(country_link, driver)
        all_data.append(data)

    driver.quit()

    df = process_data_to_df(all_data)
    print(df.head(20))
    print("\n--- Verification 1: DataFrame Info and Head ---")
    df.info()
    print("\n", df.head())

    df.to_csv("un_country_data_raw.csv", index=False)