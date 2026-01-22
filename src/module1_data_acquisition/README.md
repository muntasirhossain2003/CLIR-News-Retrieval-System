# Module 1: Data Acquisition

## Purpose

Crawl news articles from Bangladeshi news sources in both Bangla and English languages.

## Architecture

```
module1_data_acquisition/
├── crawlers/
│   ├── base_crawler.py         # Abstract base class for all crawlers
│   ├── selenium_crawler.py     # Dynamic content crawler (AJAX, infinite scroll)
│   ├── selenium_driver.py      # WebDriver manager
│   ├── selenium_buttons.py     # Button interaction strategies
│   ├── generic_crawler.py      # Static HTML crawler
│   ├── bangla_crawlers.py      # 6 Bangla news source configurations
│   └── english_crawlers.py     # 8 English news source configurations
├── utils.py                    # Text cleaning & date parsing utilities
└── generate_metadata.py        # Metadata extraction & CSV generation
```

## Components

### Base Crawler

- **File**: `base_crawler.py`
- **Class**: `BaseCrawler` (Abstract)
- **Features**:
  - HTTP request handling with retry logic
  - Automatic encoding detection (UTF-8 for Bangla)
  - MD5-based unique filename generation
  - JSON data storage with timestamps
  - Error handling and logging

### Selenium Crawler

- **File**: `selenium_crawler.py`
- **Class**: `SeleniumCrawler`
- **Handles**:
  - AJAX "Load More" buttons
  - Infinite scroll pagination
  - Numbered pagination (?page=1, ?page=2)
  - JavaScript-rendered content
- **Features**:
  - Duplicate detection
  - Progress tracking
  - Configurable max attempts per category

### News Sources

**Bangla (6 sources):**

- Prothom Alo (11 categories)
- Ittefaq (9 categories)
- Bangla Tribune (12 categories)
- Dhaka Post (9 categories)
- Samakal (8 categories)
- Jugantor (14 categories)

**English (8 sources):**

- Daily Star (12 categories)
- New Age (8 categories)
- Prothom Alo English (8 categories)
- Dhaka Tribune (11 categories)
- Financial Express (10 categories)
- NTV Bangladesh (8 categories)
- UNB (10 categories)

## Data Format

Each article stored as JSON:

```json
{
  "url": "https://example.com/article",
  "language": "bangla|english",
  "source": "source_name",
  "title": "Article Title",
  "body": "Full article text",
  "date": "Publication date",
  "crawled_at": "2026-01-22 10:30:00"
}
```

## Directory Structure

```
data/raw/
├── bangla/
│   ├── prothom_alo/
│   ├── ittefaq/
│   ├── bangla_tribune/
│   ├── dhaka_post/
│   ├── samakal/
│   └── jugantor/
└── english/
    ├── daily_star/
    ├── new_age/
    ├── prothom_alo/
    ├── dhaka_tribune/
    ├── financial_express/
    ├── ntv_bd/
    └── unb/
```

## Usage

### Crawl All Sources

```bash
python main.py crawl --limit 50
```

### Crawl Bangla Only

```bash
python main.py crawl --lang bangla --limit 100
```

### Crawl English Only

```bash
python main.py crawl --lang english --limit 100
```

### Crawl Specific Source

```bash
python main.py crawl --source daily_star --limit 200
```

### Generate Metadata CSV

```bash
python src/module1_data_acquisition/generate_metadata.py
```

## Output

- **Raw Articles**: `data/raw/{language}/{source}/`
- **Metadata CSV**: `data/metadata.csv`
- **Logs**: `logs/crawler.log`

## Statistics

Current collection: **5,194 documents** (2,589 Bangla + 2,605 English)

## Dependencies

```
requests==2.31.0
beautifulsoup4==4.12.3
selenium==4.18.1
webdriver-manager==4.0.1
pandas==2.2.0
dateparser==1.2.0
lxml==5.1.0
tqdm==4.66.1
```
