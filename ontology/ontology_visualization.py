# Python
import os
import json
import networkx as nx
import matplotlib.pyplot as plt

def build_ontology(root_dir):
    ontology = {}
    for category in os.listdir(root_dir):
        cat_path = os.path.join(root_dir, category)
        if os.path.isdir(cat_path):
            ontology[category] = {}
            for subcat in os.listdir(cat_path):
                subcat_path = os.path.join(cat_path, subcat)
                if os.path.isdir(subcat_path):
                    images = [img for img in os.listdir(subcat_path) if img.endswith(('.jpg', '.png'))]
                    ontology[category][subcat] = images
    return ontology

def visualize_general_ontology(ontology, samples_per_folder=2, samples_per_image=2):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    root = "General Category"
    G.add_node(root)

    # Get up to 4 subcategories
    subcategories = list(ontology.keys())[:4]
    folder_nodes = []
    image_nodes = []
    for subcat_idx, subcat in enumerate(subcategories):
        G.add_node(subcat)
        G.add_edge(root, subcat)
        # Get up to 2 folders for each subcategory
        folders = list(ontology[subcat].keys())[:samples_per_folder]
        for folder_idx, folder in enumerate(folders):
            folder_name = f"{subcat} → video {folder_idx+1}"
            folder_nodes.append(folder_name)
            G.add_node(folder_name)
            G.add_edge(subcat, folder_name)
            # Get up to samples_per_image images for each folder
            images = ontology[subcat][folder][:samples_per_image]
            for img_idx, img in enumerate(images):
                img_node = f"{subcat} → video {folder_idx+1} → frame {img_idx+1}"
                image_nodes.append(img_node)
                G.add_node(img_node)
                G.add_edge(folder_name, img_node)

    plt.figure(figsize=(16, 12))
    # Build shells for shell_layout: root, subcategories, folders, images
    shells = [[root], subcategories, folder_nodes, image_nodes]
    pos = nx.shell_layout(G, shells)

    # Assign colors for each level
    node_colors = []
    for node in G.nodes():
        if node == root:
            node_colors.append('#ff6666')  # red for root
        elif node in subcategories:
            node_colors.append('#ffcc66')  # orange for subcategories
        elif node in folder_nodes:
            node_colors.append('#66b3ff')  # blue for folders
        elif node in image_nodes:
            node_colors.append('#99ff99')  # green for images
        else:
            node_colors.append('#cccccc')  # fallback

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1800,
        font_size=13,
        font_weight='bold',
        node_color=node_colors,
        edge_color='#333333',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=18,
        linewidths=2
    )
    plt.title("General Ontology Hierarchical Graph", fontsize=18, fontweight='bold', color='#333333')
    plt.tight_layout()
    plt.savefig('general_ontology_graph.png')
    print("General ontology hierarchical graph saved as 'general_ontology_graph.png'")

# Usage

# Побудова онтології
root_dir = r'/ExtructedFrames extended'
ontology = build_ontology(root_dir)

# Збереження у форматі JSON
with open('ontology_for visualization.json', 'w', encoding='utf-8') as f:
    json.dump(ontology, f, ensure_ascii=False, indent=2)

visualize_general_ontology(ontology, samples_per_folder=2, samples_per_image=2)
