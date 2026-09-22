import csv
import random
from faker import Faker

fake = Faker()

CATEGORIES = ["Jackets", "Shoes", "Backpacks", "Tents", "Sleeping Bags",
              "Headlamps", "Water Bottles", "Hiking Poles", "Gloves", "Socks"]

BRANDS = ["TrailMaster", "PeakGear", "NorthRidge", "SummitCo", "WildPath",
          "AlpineEdge", "BaseCamp", "RidgeLine", "StormGuard", "TerraFlex"]

ADJECTIVES = ["waterproof", "lightweight", "durable", "breathable",
              "insulated", "packable", "rugged", "quick-dry", "windproof"]

def generate_product(pid):
    category = random.choice(CATEGORIES)
    brand = random.choice(BRANDS)
    adjective = random.choice(ADJECTIVES)
    title = f"{brand} {adjective.capitalize()} {category[:-1] if category.endswith('s') else category}"
    description = (
        f"{adjective.capitalize()} {category.lower()} from {brand}, "
        f"designed for {random.choice(['hiking', 'camping', 'trekking', 'backpacking', 'mountaineering'])}. "
        f"{fake.sentence(nb_words=12)}"
    )
    price = round(random.uniform(9.99, 349.99), 2)
    stock = random.randint(0, 500)
    rating = round(random.uniform(2.5, 5.0), 1)
    return [pid, title, description, category, brand, price, stock, rating]

def main(n=8000):
    with open("catalog.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "description", "category", "brand", "price", "stock", "rating"])
        for i in range(1, n + 1):
            writer.writerow(generate_product(i))
    print(f"Generated {n} products into catalog.csv")

if __name__ == "__main__":
    main()