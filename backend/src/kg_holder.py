_kg_instance = None

def set_kg(kg_instance):
    global _kg_instance
    _kg_instance = kg_instance

def get_kg():
    global _kg_instance
    return _kg_instance