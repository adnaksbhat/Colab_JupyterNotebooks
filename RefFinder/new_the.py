from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from flask_caching import Cache
from elasticsearch import Elasticsearch
import json, filters
import re
import spacy
from flashtext import KeywordProcessor
from nltk.stem import WordNetLemmatizer
from symspellpy import SymSpell
import logging
from dateutil import parser
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.ERROR)






# Initialize Flask app and API
app = Flask(__name__)
api = Api(app)
cors = CORS(app)
# Initialize Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})






ec = Elasticsearch(hosts=["http://localhost:9200"], request_timeout=30)






# Initialize SymSpell
sym_spell = SymSpell()
dict_path = r"C:\Users\chanska\Documents\BH_workspace\RefFinder\Reference Finder Project\Code_v2\frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(dict_path, 0, 1)

# Initialize Spacy and WordNet Lemmatizer
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# Load keyword dictionaries
with open(r"C:\Users\chanska\Documents\BH_workspace\RefFinder\Reference Finder Project\Code_v2\keys.json") as file:
    keyword_dict = json.load(file)
with open(r"C:\Users\chanska\Documents\BH_workspace\RefFinder\Reference Finder Project\Code_v2\indices.json") as f:
    index_dict = json.load(f)
    # Load synonym dictionary
with open(r"C:\Users\chanska\Documents\BH_workspace\RefFinder\Reference Finder Project\Code_v2\synonyms.json") as f:
    synonym_dict = json.load(f)

domain_stop_words = set(["project", "projects", "execute", "executed", "machines", "machine", "compressors", "compressor", "consist",
                         "consisting", "configuration", "config", "composition", "high", "height", "location", "remote", "manifold",
                         "valve", "valves", "control valves", "servovalves", "cables", "environment", "package", "packages", "motors",
                         "level", "required", "list", "air", "service", "certification", "lubrication", "type", "pads", "bearing",
                         "bearings", "arrangement", "suction", "discharge", "speed"])


stop_words = nlp.Defaults.stop_words





def expand_query_with_synonyms(query: str, synonym_dict: Dict[str, List[str]]) -> str:
    """Expands the query with synonyms from the synonym dictionary, handling multi-word synonyms."""
    expanded_terms = []
    query_terms = query.split()
    i = 0
    while i < len(query_terms):
        term = query_terms[i].lower()
        found_synonym = False
        # Check for multi-word synonyms
        for syn_key, syn_values in synonym_dict.items():
            syn_key_terms = syn_key.split()
            if len(syn_key_terms) > 1 and i + len(syn_key_terms) <= len(query_terms):
                if query_terms[i:i + len(syn_key_terms)] == [t.lower() for t in syn_key_terms]:
                    expanded_terms.append(f"({' '.join([syn_key] + syn_values)})")
                    i += len(syn_key_terms)
                    found_synonym = True
                    break
        if not found_synonym:
            if term in synonym_dict:
                # Check if the synonym key is a multi-word term
                if len(synonym_dict[term][0].split()) > 1:
                    expanded_terms.append(f"({' '.join([term] + synonym_dict[term])})")
                else:
                    expanded_terms.append(f"({' '.join([term] + synonym_dict[term])})")
            else:
                expanded_terms.append(term)
            i += 1
    return " ".join(expanded_terms)






def format_filters(filter_dict):
    temp_list = []
    for key, values in filter_dict.items():
        if type(values) == list:
            temp_list.append({"terms": {str(key).upper() + (".keyword"): values}})
        elif type(values) == dict:
            temp_list.append({"range": {str(key).upper(): values}})
        elif type(values) == str:
            try:
                date_obj = parser.parse(values)
                temp_list.append({"range": {str(key).upper(): {"gte": date_obj.isoformat()}}})
            except ValueError:
                pass
    return temp_list








def sort_highlights(highlights, sort_list):
    sort_list = [re.sub(r'[\*\(\)]', '', _) for _ in sort_list]
    sort_list = [item for item in sort_list if item != ""]
    highlight_fields = list(highlights.keys())
    fields = [field_name for partial_field in sort_list for field_name in highlight_fields if partial_field in field_name]
    d_sorted = {ky: highlights[ky] for ky in fields}
    [d_sorted.update({key: highlights[key]}) for key in highlights.keys() if key not in d_sorted]
    return d_sorted









def generate_wildcard_variations(term):
    """Generates wildcard variations of a term, including prefix and suffix."""
    variations = []
    #variations.append(f"*{term}*")  # Contains
    #variations.append(f"{term}*")  # Starts with
    #variations.append(f"*{term}")  # Ends with
    return variations










def remove_stops(query):
    # Text Normalization
    query = query.lower()
    query = re.sub(r'[^\w\s/<>+\-=.%]', '', query)
    query = re.sub(r'\S*@\S*\s?', '', query)
    query = re.sub(r'\s+', ' ', query)
    query = query.replace("+", "\+")

    # Extract keywords from query
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_dict(keyword_dict)
    keywords1 = keyword_processor.extract_keywords(query)
    keywords = ["*" + keyword.upper() + "*" for keyword in keywords1]
    keywords_weighted = [keyword + "^50" for keyword in keywords]

    keykey = set([item.lower() for item in keywords1])

    # Replace keywords in query
    query_new = keyword_processor.replace_keywords(query)

    # Tokenization
    doc = nlp(query)

    original_query_terms = [token.text for token in doc]


    filtered_tokens = []
    for token in doc:
        token_text_lower = token.text.lower()
        if token_text_lower not in stop_words and token_text_lower not in domain_stop_words and len(token_text_lower) > 1 and not token.is_punct and not token.is_space:
            # Check if the token is part of a multi-word keyword
            is_part_of_keyword = False
            for keyword in keykey:
                if " " in keyword and token_text_lower in keyword.split():
                    is_part_of_keyword = True
                    break
            if not is_part_of_keyword:
                filtered_tokens.append(token.lemma_)
        
    filtered_sentence = " ".join(filtered_tokens).strip()


    # Define columns
    columns = ["*COUNTRY*", "*PLANT NAME*", "*TRAIN CONFIGURATION*",
               "*YEAR*", "*PROJECT NAME*", "*EPC*", "*INSTALLATION TYPE2*", "*INSTALLATION*", "*COMPRESSOR*",
               "*MANUFACTURER*", "*DESCRIPTION*", "*N P JOB NUMBER*", "*JOB NUMBER*", "*JOB*", "*ENVIRONMENT*", "*JOB ID*",
               "*MOTORS*", "*ELECTRIC*", "*GENERAL*", "*INSTRUMENTS*", "*DOCUMENT*", "*VENDOR*" ,"*CUSTOMER NAME*", "*PLANT LOCATION*"]



    # Update columns with keywords
    if keywords:
        columns_weighted = columns + keywords_weighted + ["*NOTES^0"]
        columns = keywords + columns
    else:
        columns.append("*")
        columns_weighted = columns

    # Extract indices
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_dict(index_dict)
    indices = keyword_processor.extract_keywords(query)
    index_boost = [{index: 200} for index in indices]


    # Initialize highlight fields
    highlight_fields = {}
    for column in columns:
        highlight_fields[column] = {}
    if not columns:
        highlight_fields["*"] = {}




    print("Original Query ---------->", query)
    print("original_query_terms: ---------->", original_query_terms)
    print("Columns found (extract keywords from json): ---------->", keywords1)
    #print("keykey (lower case) ---------->", keykey)
    print("filtered_tokens: ---------->", filtered_tokens)
    print("filtered_sentence: ----------> ", filtered_sentence)
    #print("highlight_fields: ----------> ", highlight_fields)
    # print("Columns:", columns)
    # print("Columns Weighted:", columns_weighted)
    # print("Index Boost:", index_boost)

   

    return filtered_sentence, columns_weighted, highlight_fields, columns, index_boost, original_query_terms
















































class Search2(Resource):
    @cache.cached(timeout=3600, key_prefix=lambda: request.args.get('search'))
    def get(self):
        args = request.args
        index = args.get('index_name')
        query_string = args.get('search')
        pgno = int(args.get('pgno'))

        # Spelling correction
        suggestions = sym_spell.lookup_compound(query_string.lower(), max_edit_distance=2, ignore_non_words=True,
                                               ignore_term_with_digits=True)
        suggest = suggestions[0].term if suggestions else "NA"

        # Process query
        query_string, fields, highlight_fields, columns, index_boost, original_query_terms = remove_stops(query_string)

        query_string = expand_query_with_synonyms(query_string, synonym_dict)

        print("Query after expansion with synonyms ---------->", query_string)



        # Determine filter aggregation
        filters_agg = filters.filters[int(index)] if index.isdigit() and int(index) < len(
            filters.filters) else {'indices': {'terms': {'field': '_index'}}}
        
        must_not_clauses = []
        # Handle "without", "not", or "non" scenarios more efficiently
        negative_indicators = ["without", "not", "non", "not", "no", "none"]
        i = 0
        while i < len(original_query_terms):
            term = original_query_terms[i]
            if term.lower() in negative_indicators and i + 1 < len(original_query_terms):
                excluded_term = original_query_terms[i + 1]
                
                # Check if the next term is part of a multi-word key
                found_multi_word_key = False
                for key, values in keyword_dict.items():
                    key_parts = key.split()
                    if len(key_parts) > 1 and excluded_term.lower() in [part.lower() for part in key_parts]:
                        # Check if the subsequent terms match the rest of the multi-word key
                        if i + 1 + len(key_parts) <= len(original_query_terms):
                            if all(original_query_terms[i + 1 + j].lower() == key_parts[j].lower() for j in range(len(key_parts))):
                                excluded_term = key
                                i += len(key_parts)
                                found_multi_word_key = True
                                break
                
                if not found_multi_word_key:
                    # Find the field where the term should not be present more efficiently
                    field_to_exclude = next((key for key, values in keyword_dict.items() if excluded_term.lower() in [item.lower() for item in values]), "")
                else:
                    field_to_exclude = next((key for key, values in keyword_dict.items() if excluded_term.lower() in [item.lower() for item in values]), "")

                if field_to_exclude:
                    must_not_clauses.append(
                        {
                            "match": {
                                f"*{field_to_exclude}*": {
                                    "query": excluded_term,
                                    "analyzer": "custom_analyzer"
                                }
                            }
                        }
                    )
                else:
                    must_not_clauses.append(
                        {
                            "query_string": {
                                "query": excluded_term,
                                "fields": fields,
                                "analyzer": "custom_analyzer",
                                "lenient": True,
                                "default_operator": "or"
                            }
                        }
                    )
                # Skip the next term since it's already processed as the excluded term
                
                i += 2  # Skip the negative indicator and the excluded term
            else:
                i += 1
            
        # --- Modified Elasticsearch Query ---
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_string,
                                "fields": fields,
                                "type": "cross_fields",  # Changed to cross_fields
                                "operator": "or",  # Added operator
                                "lenient": True,
                                "analyzer": "custom_analyzer"
                            }
                        }
                    ],
                    "should": [
                       
                        {
                            "multi_match":{
                               "query":query_string,
                               "fields": fields,
                               "type": "cross_fields",
                               "operator": "and",
                               "lenient": True,
                               "analyzer": "custom_analyzer",
                               "boost": 40
                            }
                        },
                        {
                            "multi_match":{
                               "query":query_string,
                               "fields": fields,
                               "type": "most_fields",
                               "operator": "and",
                               "lenient": True,
                               "analyzer": "custom_analyzer",
                               "boost": 10
                            }
                        },
                        # Add a wildcard query for partial matches
                        {
                            "query_string": {
                                "query": " OR ".join([variation for term in query_string.split() for variation in generate_wildcard_variations(term)]),
                                "fields": fields,
                                "boost": 10,
                                "analyzer": "custom_analyzer",
                                "lenient": True,
                                "default_operator": "or"
                            }
                        }
                    
                    ],
                    
                    "must_not": must_not_clauses,
                    "filter": format_filters(request.args.get('filter', {}))
                }
            },
            "size": 10,
            "from": pgno * 10,
            "highlight": {
                "pre_tags": ["|~S~|"],
                "post_tags": ["|~E~|"],
                "fields": {"*": {}},
                "order": "none"
            },
            "aggs": filters_agg,
            "indices_boost": index_boost
        }
        # --- End of Modified Elasticsearch Query ---

        # Execute search
        result = ec.search(index=index, body=query)

        # Process highlights and aggregations
        if "hits" in result and "hits" in result["hits"]:
            for col in result['hits']['hits']:
              if "highlight" in col:
                if col.get("highlight"):
                    col['highlight'] = sort_highlights(col['highlight'], columns)

        if "aggregations" in result:
            for k, v in result['aggregations'].items():
                result['aggregations'][k] = v.get('buckets', [v])
       
       
        result.body["spell_check_suggestion"] = suggest

        return result.body

api.add_resource(Search2, '/Search2')

if __name__ == '__main__':
    app.run()
