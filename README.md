# CLIR Project: Cross-Lingual Information Retrieval System

## 📋 Project Overview

This project implements a **Cross-Lingual Information Retrieval (CLIR)** system for Bangla-English news articles. The system is designed to crawl, process, index, and retrieve news articles from multiple Bangladeshi news sources in both Bengali and English languages.

### Current Status: Module 1 - Data Acquisition (✅ Completed)


## 📊 Current Implementation: Data Acquisition Module

### ✅ What Has Been Accomplished

#### 1. **Web Crawling Infrastructure**

I have built a robust, scalable web crawling system with the following components:

##### **Base Crawler Architecture** ([base_crawler.py](src/module1_data_acquisition/crawlers/base_crawler.py))

- Abstract base class for all crawlers
- Handles HTTP requests with retry logic and respectful crawling delays
- Automatic encoding detection for Bangla/English content
- MD5-based unique filename generation to avoid duplicates
- Comprehensive logging system with UTF-8 support
- JSON-based data storage with timestamping
- Built-in error handling and recovery mechanisms

##### **Selenium-Based Dynamic Crawler** ([selenium_crawler.py](src/module1_data_acquisition/crawlers/selenium_crawler.py))

A sophisticated crawler for modern JavaScript-heavy news websites with:

- **Multiple Pagination Strategies:**
  - AJAX "Load More" button handling
  - Infinite scroll detection and management
  - Numbered pagination (`?page=1`, `?page=2`, etc.)
- **Smart Detection:**
  - Duplicate article prevention
  - Progress tracking to avoid infinite loops
  - Configurable maximum attempts per category
- **Driver Management** ([selenium_driver.py](src/module1_data_acquisition/crawlers/selenium_driver.py)):
  - Headless Chrome browser automation
  - Anti-detection measures (stealth mode)
  - Automatic driver installation via webdriver-manager
  - Memory-efficient page loading
- **Button Strategy System** ([selenium_buttons.py](src/module1_data_acquisition/crawlers/selenium_buttons.py)):
  - Intelligent button click attempts with retries
  - Scroll-based element visibility handling
  - Infinite scroll with progress monitoring

##### **Generic Static Crawler** ([generic_crawler.py](src/module1_data_acquisition/crawlers/generic_crawler.py))

- For traditional, non-JavaScript news sites
- Simple HTML parsing with BeautifulSoup
- Link extraction and pagination support

#### 2. **News Source Coverage**

##### **Bangla News Sources** (6 sources) - [bangla_crawlers.py](src/module1_data_acquisition/crawlers/bangla_crawlers.py)

1. **Prothom Alo** (`prothom_alo`) - AJAX load more, 11 categories
2. **The Daily Ittefaq** (`ittefaq`) - AJAX pagination, 9 categories
3. **Bangla Tribune** (`bangla_tribune`) - AJAX load more, 12 categories
4. **Dhaka Post** (`dhaka_post`) - Infinite scroll, 9 categories
5. **Samakal** (`samakal`) - AJAX pagination, 8 categories
6. **Jugantor** (`jugantor`) - AJAX pagination, 14 categories

##### **English News Sources** (8 sources) - [english_crawlers.py](src/module1_data_acquisition/crawlers/english_crawlers.py)

1. **The Daily Star** (`daily_star`) - Numbered pagination, 12 categories
2. **New Age** (`new_age`) - Numbered pagination, 8 categories
3. **Daily Observer** (`daily_observer`) - Static site
4. **Prothom Alo English** (`prothom_alo`) - Infinite scroll, 8 categories
5. **Dhaka Tribune** (`dhaka_tribune`) - AJAX pagination, 11 categories
6. **Financial Express** (`financial_express`) - Numbered pagination, 10 categories
7. **NTV Bangladesh** (`ntv_bd`) - AJAX pagination, 8 categories
8. **United News of Bangladesh (UNB)** (`unb`) - AJAX pagination, 10 categories

**Total: 14 News Sources** across both languages

#### 3. **Data Structure & Storage**

##### **Article Data Model**

Each article is stored as a JSON file containing:

```json
{
  "url": "https://example.com/article",
  "language": "bangla|english",
  "source": "source_name",
  "title": "Article Title",
  "body": "Full article text content",
  "date": "Publication date (if available)",
  "crawled_at": "2025-12-18 06:37:25"
}
```

##### **Directory Structure**

```
data/
├── raw/
│   ├── bangla/
│   │   ├── prothom_alo/
│   │   │   ├── 803452_2cfd948f.json
│   │   │   ├── 812787_7eedad87.json
│   │   │   └── ...
│   │   ├── ittefaq/
│   │   ├── bangla_tribune/
│   │   ├── dhaka_post/
│   │   ├── samakal/
│   │   └── jugantor/
│   └── english/
│       ├── daily_star/
│       ├── new_age/
│       ├── daily_observer/
│       ├── prothom_alo/
│       ├── dhaka_tribune/
│       ├── financial_express/
│       ├── ntv_bd/
│       └── unb/
└── metadata.csv
```

#### 4. **Metadata Generation System** ([generate_metadata.py](src/module1_data_acquisition/generate_metadata.py))

Automated metadata extraction and cleaning:

- Scans all JSON files in the data directory
- Extracts key metadata fields
- **Data Cleaning Operations:**
  - Removes duplicate articles (by URL)
  - Filters out entries with missing titles or URLs
  - Sorts by language and source
- Generates comprehensive CSV file ([metadata.csv](data/metadata.csv)) with:
  - Filename, language, source, URL, title, date, crawl timestamp, filepath
- **Statistics:** Currently contains **5,634 articles** across both languages

#### 5. **Utility Functions** ([utils.py](src/module1_data_acquisition/utils.py))

- `clean_text()`: Removes extra whitespace and normalizes text
- `parse_date()`: Flexible date parsing with dateparser library

#### 6. **Command-Line Interface** ([main.py](main.py))

Flexible CLI for crawling operations:

```bash
# Crawl all sources (default 50 articles per site)
python main.py

# Crawl only Bangla sources
python main.py --lang bangla

# Crawl only English sources
python main.py --lang english

# Specify number of articles per site
python main.py --limit 100

# Crawl specific source
python main.py --source bangla_tribune --limit 200

# Combine options
python main.py --lang english --limit 150
```

**Features:**

- Language selection (bangla/english/all)
- Article limit per source
- Single source targeting
- Error handling and logging
- Progress tracking

---

## 🔧 Technical Stack

### **Core Technologies**

- **Python 3.x** - Main programming language
- **Beautiful Soup 4** - HTML parsing
- **Selenium** - Dynamic content & JavaScript handling
- **Requests** - HTTP requests
- **Pandas** - Data manipulation & CSV handling

### **Dependencies** ([requirements.txt](requirements.txt))

#### Web Crawling & Parsing

- `requests==2.31.0` - HTTP library
- `beautifulsoup4==4.12.3` - HTML/XML parsing
- `lxml==5.1.0` - Fast XML/HTML processing
- `selenium==4.18.1` - Browser automation
- `webdriver-manager==4.0.1` - Automatic driver management

#### Data Processing

- `pandas==2.2.0` - Data analysis & manipulation
- `dateparser==1.2.0` - Flexible date parsing
- `tqdm==4.66.1` - Progress bars

---

## 📈 Dataset Statistics

### **Current Collection (As of December 27, 2025)**

- **Total Articles:** 5,634 unique articles
- **Languages:** Bangla & English
- **Sources:** 14 different news outlets
- **Data Quality:**
  - All articles have titles and URLs
  - Duplicates removed
  - UTF-8 encoded for Bangla support
  - Timestamped for tracking

### **Article Distribution by Source**

Varies by source availability and category coverage. View detailed breakdown:

```bash
python src/module1_data_acquisition/generate_metadata.py
```

---

## 🚀 Installation & Setup

### **1. Clone the Repository**

```bash
cd clir-project
```

### **2. Create Virtual Environment (Recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Setup Chrome Driver**

The system automatically downloads and manages Chrome drivers via `webdriver-manager`. Ensure you have Chrome browser installed.

---

## 💻 Usage Guide

### **Basic Crawling**

#### Crawl All Sources

```bash
python main.py --lang all --limit 50
```

#### Crawl Bangla Sources Only

```bash
python main.py --lang bangla --limit 100
```

#### Crawl Specific Source

```bash
python main.py --source daily_star --limit 200
```

### **Generate/Update Metadata**

After crawling, generate the metadata CSV:

```bash
python src/module1_data_acquisition/generate_metadata.py
```

This will:

- Scan all JSON files
- Remove duplicates
- Clean missing data
- Generate `data/metadata.csv`
- Display statistics by source

### **Logging**

All crawl activities are logged to:

```
logs/crawler.log
```

---

## 🏗️ System Architecture

### **Module 1: Data Acquisition (✅ Current)**

```
┌─────────────────────────────────────────────┐
│           Main CLI (main.py)                │
├─────────────────────────────────────────────┤
│   ┌─────────────┐     ┌──────────────┐     │
│   │   Bangla    │     │   English    │     │
│   │  Crawlers   │     │   Crawlers   │     │
│   └──────┬──────┘     └──────┬───────┘     │
│          │                   │              │
│   ┌──────▼───────────────────▼───────┐     │
│   │     Selenium Crawler             │     │
│   │  ┌─────────┐  ┌──────────────┐  │     │
│   │  │ Driver  │  │    Button     │  │     │
│   │  │ Manager │  │   Strategy    │  │     │
│   │  └─────────┘  └──────────────┘  │     │
│   └──────────────────────────────────┘     │
│   ┌──────────────────────────────────┐     │
│   │     Base Crawler (Abstract)      │     │
│   └──────────────────────────────────┘     │
├─────────────────────────────────────────────┤
│              Data Storage                   │
│   ┌──────────┐          ┌────────────┐     │
│   │   JSON   │   ───>   │ metadata   │     │
│   │  Files   │          │    .csv    │     │
│   └──────────┘          └────────────┘     │
└─────────────────────────────────────────────┘
```

### **Design Patterns Used**

1. **Abstract Base Class Pattern** - `BaseCrawler` provides interface
2. **Strategy Pattern** - `ButtonClickStrategy` for different pagination types
3. **Factory Pattern** - `get_bangla_crawlers()` and `get_english_crawlers()`
4. **Singleton Pattern** - Driver management
5. **Template Method Pattern** - `crawl()` method workflow

---

## 🎓 Key Features & Innovations

### **1. Robust Crawling Strategies**

- Handles static HTML, AJAX, infinite scroll, and paginated sites
- Automatic retry with exponential backoff
- Respectful crawling with configurable delays
- Anti-detection measures for Selenium

### **2. Bangla Language Support**

- UTF-8 encoding throughout
- Proper handling of Bangla text in logs and files
- Bangla date parsing support

### **3. Data Quality Assurance**

- Duplicate detection and removal
- Missing data filtering
- URL validation
- Content normalization

### **4. Scalability**

- Modular crawler design for easy addition of new sources
- Efficient file storage with MD5 hashing
- Parallel processing capability (can be extended)

### **5. Monitoring & Debugging**

- Comprehensive logging system
- Progress indicators with `tqdm`
- Detailed error messages
- Per-source statistics

---

## 📁 Project Structure Breakdown

```
clir-project/
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── data/                            # All collected data
│   ├── metadata.csv                 # Centralized metadata (5,634 articles)
│   └── raw/                         # Raw article JSON files
│       ├── bangla/                  # Bangla news sources
│       │   ├── prothom_alo/
│       │   ├── ittefaq/
│       │   ├── bangla_tribune/
│       │   ├── dhaka_post/
│       │   ├── samakal/
│       │   └── jugantor/
│       └── english/                 # English news sources
│           ├── daily_star/
│           ├── new_age/
│           ├── daily_observer/
│           ├── prothom_alo/
│           ├── dhaka_tribune/
│           ├── financial_express/
│           ├── ntv_bd/
│           └── unb/
│
├── logs/                            # Crawling logs
│   └── crawler.log
│
└── src/                             # Source code
    └── module1_data_acquisition/
        ├── __init__.py
        ├── generate_metadata.py     # Metadata extraction & cleaning
        ├── utils.py                 # Utility functions
        └── crawlers/
            ├── __init__.py
            ├── base_crawler.py      # Abstract base crawler
            ├── selenium_crawler.py  # Main Selenium crawler
            ├── selenium_driver.py   # WebDriver management
            ├── selenium_buttons.py  # Button click strategies
            ├── generic_crawler.py   # Static site crawler
            ├── bangla_crawlers.py   # Bangla source configs
            └── english_crawlers.py  # English source configs
```

_Last Generated: December 27, 2025_
