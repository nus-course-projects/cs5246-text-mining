import re
import unicodedata
import pandas as pd
import spacy
import geonamescache  # type: ignore
import pycountry
from tqdm import tqdm
from utils.dataset.articles import ArticlesDataset


class LocationExtractor:
    def __init__(self, articles: ArticlesDataset):
        self.articles = articles

        # Build combined dataframe with article content and labels
        contents = []
        print("Building dataframe...")
        for idx in tqdm(range(len(articles))):
            _, content, _ = articles[idx]
            contents.append(content)
        self.df = pd.concat([articles.filtered_rows.reset_index(drop=True), articles.filtered_labels.reset_index(drop=True)], axis=1)
        self.df['content'] = contents

        self.nlp = spacy.load("en_core_web_lg")

        # Load geonames data
        self.gc = geonamescache.GeonamesCache()
        self.cities = self.gc.get_cities()
        self.countries = self.gc.get_countries()

        # Build helper mappings
        self.country_name_mapping = self.build_country_name_mapping()

         # US-specific city to state mapping
        self.city_to_state = {
            self.normalize_text(city_info["name"]): city_info["admin1code"]
            for city_info in self.cities.values()
            if city_info["countrycode"] == "US"
        }
        
        # City to country mapping
        self.city_to_country = {
            self.normalize_text(city_info['name']): self.gc.get_countries()[city_info['countrycode']]['name'].lower()
            for city_info in self.cities.values()
        }

        # US state to country mapping (for subdivision resolution)
        self.state_to_country = {
            self.normalize_text(subdivision.name): subdivision.country.name.lower()
            for subdivision in pycountry.subdivisions
            if subdivision.country_code == "US" and subdivision.type == "State"
        }

        # List of normalized country names
        self.country_names = [self.normalize_text(info['name']) for code, info in self.gc.get_countries().items()]

    # Normalize text to lowercase ASCII (removes accents, special chars)
    def normalize_text(self, text: str):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').lower()

    # Normalize and remove "the " prefix from country names
    def clean_country_name(self, name: str):
        name = self.normalize_text(name)
        if name.startswith('the '):
            name = name[4:]
        return name

    # Build a mapping of official and alternate country names to standard name
    def build_country_name_mapping(self):
        mapping = {}
        for country in pycountry.countries:
            official_name = self.normalize_text(country.name)
            mapping[official_name] = official_name

            if hasattr(country, 'official_name'):
                official = self.normalize_text(country.official_name)
                mapping[official] = official_name

            if hasattr(country, 'common_name'):
                common = self.normalize_text(country.common_name)
                mapping[common] = official_name
        return mapping

    # Convert any alias to standard name using cleaned format
    def map_country_alias(self, name):
        cleaned = self.clean_country_name(name)
        return self.country_name_mapping.get(cleaned, cleaned)

    # Extract potential cities, states, and countries from the given text using NER and lookups
    def extract_from_text(self, text):
        if not text:
            return {"cities": [], "states": [], "countries": []}
        # Handle uppercase-heavy text like all-caps headlines
        words = text.split()
        if len(words) > 0:
            capitalized_words = sum(1 for word in words if word[0].isupper())
            if (capitalized_words / len(words)) > 0.8:
                text = text.lower()
        
        # NER
        doc = self.nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "NORP", "LOC"]]
        candidates = set(locations)
        cities_found = []
        states_found = []
        countries_found = []

        for candidate in candidates:
            # Clean and normalize candidate
            candidate_clean = re.sub(r"'s\b", "", candidate)
            candidate_clean = re.sub(r"[^\w\s]", "", candidate_clean)
            normalized = self.normalize_text(candidate_clean)
            normalized = self.correct_partial_location(normalized)
            mapped_country_name = self.map_country_alias(normalized)

            # Match against known countries/states/cities
            if mapped_country_name in self.country_names:
                countries_found.append(mapped_country_name)
            elif normalized in self.state_to_country:
                country = self.normalize_text(self.state_to_country[normalized])
                states_found.append(candidate)
                countries_found.append(country)
            elif normalized in self.city_to_country:
                country = self.normalize_text(self.city_to_country[normalized])
                cities_found.append(candidate)
                countries_found.append(country)

                # Resolve US state from city if possible
                if country == "united states":
                    state_code = self.city_to_state.get(normalized)
                    if state_code:
                        subdivision = pycountry.subdivisions.get(code=f"US-{state_code}")
                        if subdivision:
                            states_found.append(subdivision.name)
        return {
            "cities": list(set(cities_found)),
            "states": list(set(states_found)),
            "countries": list(set(countries_found)),
        }

    # If full location name isn't found, try resolving from parts (e.g., possessives or substrings)
    def correct_partial_location(self, candidate: str):
        candidate_clean = re.sub(r"'s\b", "", candidate)
        candidate_clean = re.sub(r"[^\w\s]", "", candidate_clean)
        normalized = self.normalize_text(candidate_clean)

        mapped_country = self.map_country_alias(normalized)
        if (mapped_country in self.country_names) or (normalized in self.city_to_country) or (normalized in self.state_to_country):
            return normalized
        
        # Trying partial matches by splitting on spaces
        words = normalized.split()
        for word in words:
            if word in self.country_names or word in self.city_to_country or word in self.state_to_country:
                return word

        return candidate
    
    # Extract location from title (preferred) or content (fallback)
    def extract_location(self, title: str, content: str):
        title_result = self.extract_from_text(title)
        content_result = self.extract_from_text(content)

        final_result: dict[str, list] = {
            "cities": [],
            "states": [],
            "countries": [],
        }

        if title_result["cities"] or title_result["states"] or title_result["countries"]:
            final_result = title_result
        else:
            final_result = content_result

        return final_result

    # Soft string comparator for fuzzy matching between extracted event name & LLM label
    def build(self):
        city_match = 0
        state_match = 0
        country_match = 0
        total = len(self.df)
        def soft_compare(loc1: str | None, loc2: str | None):
            if loc1 == loc2:
                return True
            if loc1 and not loc2:
                return None
            if not loc1 and loc2:
                return False
            if loc1 and loc2:
                if loc1.lower() in loc2.lower():
                    return True
                if loc2.lower() in loc1.lower():
                    return True
                return False
            return False

        for idx, row in tqdm(self.df.iterrows(), desc="Extracting locations", total=len(self.df)):
            title = row["title"]
            content = row["content"]
            result = self.extract_location(title, content)
            extracted_city = result["cities"][0] if result["cities"] else None
            extracted_state = result["states"][0] if result["states"] else None
            extracted_country = result["countries"][0] if result["countries"] else None

            self.df.at[idx, "extracted_city"] = extracted_city
            self.df.at[idx, "extracted_state"] = extracted_state
            self.df.at[idx, "extracted_country"] = extracted_country

            city_match += 1 if soft_compare(extracted_city, row["city"]) else 0
            state_match += 1 if soft_compare(extracted_state, row["state"]) else 0
            country_match += 1 if soft_compare(extracted_country, row["country"].iloc[1]) else 0
            
            # Compute accuracy (based on non-null predictions)
            total_city = len(self.df["extracted_city"].notna())
            total_state = len(self.df["extracted_state"].notna())
            total_country = len(self.df["extracted_country"].notna())
        return city_match / total_city, state_match / total_state, country_match / total_country
