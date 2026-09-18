import re
import math
from urllib.parse import urlparse


def extract_features(url):

    parsed = urlparse(url)
    domain = parsed.netloc

    # URL length
    url_length = len(url)

    # Number of dots
    num_dots = url.count(".")

    # HTTPS
    has_https = 1 if url.startswith("https://") else 0

    # IP address
    ip_pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    has_ip = 1 if re.match(ip_pattern, domain.split(":")[0]) else 0

    # Number of subdirectories
    num_subdirs = len([x for x in parsed.path.split("/") if x])

    # Number of parameters
    num_params = len(parsed.query.split("&")) if parsed.query else 0

    # Suspicious words
    suspicious_list = [
        "login",
        "verify",
        "account",
        "update",
        "secure",
        "password",
        "bank",
        "confirm"
    ]

    suspicious_words = sum(
        word in url.lower()
        for word in suspicious_list
    )

    # Special characters
    special_char_count = len(
        re.findall(r"[@?=&_%]", url)
    )

    # Number of digits
    digits_count = sum(
        char.isdigit()
        for char in url
    )

    # Entropy
    if url:
        frequency = {}

        for char in url:
            frequency[char] = frequency.get(char, 0) + 1

        entropy = 0

        for count in frequency.values():
            probability = count / len(url)
            entropy -= probability * math.log2(probability)

    else:
        entropy = 0

    return {
        "url_length": url_length,
        "num_dots": num_dots,
        "has_https": has_https,
        "has_ip": has_ip,
        "num_subdirs": num_subdirs,
        "num_params": num_params,
        "suspicious_words": suspicious_words,
        "special_char_count": special_char_count,
        "digits_count": digits_count,
        "entropy": entropy
    }