import os
import json

def generate_ontology_from_folder(folder_path):
    ontology = {
        "domain": "військова техніка",
        "categories": []
    }

    for category_name in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category_name)
        if os.path.isdir(category_path):
            category = {
                "id": str(len(ontology["categories"]) + 1),
                "name": category_name,
                "subcategories": []
            }

            for subcategory_name in os.listdir(category_path):
                subcategory_path = os.path.join(category_path, subcategory_name)
                if os.path.isdir(subcategory_path):
                    subcategory = {
                        "id": f"{category['id']}.{len(category['subcategories']) + 1}",
                        "name": subcategory_name,
                        "images": []
                    }

                    for image_name in os.listdir(subcategory_path):
                        image_path = os.path.join(subcategory_path, image_name)
                        if os.path.isfile(image_path):
                            image = {
                                "id": f"{subcategory['id']}.{len(subcategory['images']) + 1}",
                                "filename": image_name
                            }
                            subcategory["images"].append(image)

                    category["subcategories"].append(subcategory)

            ontology["categories"].append(category)

    return ontology


folder_path = "ExtructedFrames extended"


ontology = generate_ontology_from_folder(folder_path)


output_file = "ontology.json"
if os.path.exists(output_file):
    os.remove(output_file)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(ontology, f, ensure_ascii=False, indent=4)

print(f"Онтологія успішно створена та збережена у файл {output_file}")