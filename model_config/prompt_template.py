def inject_prompt(context, question):
    prompt = f"Pouze na základě následujícího kontextu odpověz na otázku. Odpověď by neměla být delší než jeden krátký odstavec.\n\nKontext:\n{context}\n\nOtázka: {question}"
    return prompt