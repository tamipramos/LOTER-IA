import datetime
from playwright.sync_api import sync_playwright
from itertools import product
from lottery import Lottery
import helpers
import time
import config
from loteria import NumberPredictor
import os
import time

ONCE_URL = "https://www.juegosonce.es/historico-resultados-cupones-once"

def populate_db():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(ONCE_URL)

        try:
            page.wait_for_selector("button#accept", timeout=5000)
            page.click("button#accept")
        except:
            print("[!]  No cookie banner found or already accepted")

        page.wait_for_selector("select#anyo")
        page.wait_for_selector("select#mes")

        years = [opt.get_attribute("value") for opt in page.query_selector_all("select#anyo > option") if opt.get_attribute("value")][::-1]
        months = [opt.get_attribute("value") for opt in page.query_selector_all("select#mes > option") if opt.get_attribute("value")]

        for year, month in product(years, months):
            print(f"[*]  Looking for sequences -> {month}/{year}...")
            page.select_option("select#anyo", value=year)
            page.select_option("select#mes", value=month)

            page.click("body > section.cont.resultados > div.seeker > form > fieldset > div > input")

            try:
                page.wait_for_selector("div.detresult", timeout=5000)
                result_blocks = page.query_selector_all("div.detresult")

                for block in result_blocks:
                    text_raw = block.inner_text().strip()
                    if not text_raw:
                        continue
                    
                    bloques = helpers.split_results(text_raw)
                    n=0
                    for sub_text in bloques:
                        result = helpers.parse_result(sub_text)
                        result["year"] = year
                        result["month"] = month
                        helpers.save_result_to_json(data=result)
                        lottery = Lottery.from_dict(result)
                        lottery.save_to_db(config.db)
                        n += 1
                    print(f"[*] Results saved in DB: {n}")
            except Exception as e:
                continue

            time.sleep(0.5)


        browser.close()

def update_db():
    current_year = str(datetime.datetime.now().year)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(ONCE_URL)

        try:
            page.wait_for_selector("button#accept", timeout=5000)
            page.click("button#accept")
        except:
            print("[!]  No cookie banner found or already accepted")

        page.wait_for_selector("select#anyo")
        page.wait_for_selector("select#mes")

        years = [opt.get_attribute("value") for opt in page.query_selector_all("select#anyo > option") if opt.get_attribute("value")][::-1]
        months = [opt.get_attribute("value") for opt in page.query_selector_all("select#mes > option") if opt.get_attribute("value")]

        
        years = [y for y in years if y == current_year]

        for year, month in product(years, months):
            print(f"[*]  Looking for sequences -> {month}/{year}...")
            page.select_option("select#anyo", value=year)
            page.select_option("select#mes", value=month)

            page.click("body > section.cont.resultados > div.seeker > form > fieldset > div > input")

            try:
                page.wait_for_selector("div.detresult", timeout=5000)
                result_blocks = page.query_selector_all("div.detresult")

                for block in result_blocks:
                    text_raw = block.inner_text().strip()
                    if not text_raw:
                        continue
                    
                    bloques = helpers.split_results(text_raw)
                    n = 0
                    for sub_text in bloques:
                        result = helpers.parse_result(sub_text)
                        result["year"] = year
                        result["month"] = month
                        helpers.save_result_to_json(data=result)
                        lottery = Lottery.from_dict(result)
                        lottery.save_to_db(config.db)
                        n += 1
                    print(f"[*] Results saved in DB: {n}")
            except Exception:
                continue

            time.sleep(0.5)

        browser.close()

def represent_number(pos, digit):
    num = ['0'] * 5
    num[pos - 1] = str(digit)
    return ''.join(num)

def calculate_probabilities():
    def empirical_probability():
        print(f"\nEmpirical probability that a digit d is in position i")
        for i in range(1, 6):
            for d in range(0, 10):
                prob = config.db.execute_custom_query(f"""
                    SELECT
                        COUNT(*) * 1.0 / (SELECT COUNT(*) FROM lottery) AS prob
                    FROM lottery
                    WHERE substr(number, {i}, 1) = '{d}'; 
                """, ())[0]['prob']
                print(f"{represent_number(i, d)}: {prob*100:.4f} %")
    
    def probility_of_digit():
        print(f"\nProbability of digit d in any position")
        for d in range(0, 10):
            prob = config.db.execute_custom_query(f"""
                SELECT
                    COUNT(*) * 1.0 / (SELECT COUNT(*) FROM lottery) AS prob
                FROM lottery
                WHERE instr(number, '{d}') > 0;
            """, ())[0]['prob']
            print(f"{d}: {prob*100:.4f} %")

    def probability_of_digit_at_least_once():
        print(f"\nProbability that digit X appears at least once")
        for d in range(0, 10):
            prob_any = config.db.execute_custom_query(f"""
                SELECT
                    COUNT(*) * 1.0 / (SELECT COUNT(*) FROM lottery) AS prob_any_X
            FROM lottery
            WHERE instr(number, '{d}') > 0;
        """, ())[0]['prob_any_X']
        print(f"{d}: {prob_any*100:.4f} %")

    def probability_of_adjacent_digits():
        print(f"\nProbability of occurrence of the ordered pair XY as adjacent digits")
        for d in range(0, 10):
            for d2 in range(0, 10):
                prob_adj = config.db.execute_custom_query(f"""
                    SELECT
                    COUNT(*) * 1.0 / (SELECT COUNT(*) FROM lottery) AS prob_adjacent
                FROM lottery
                WHERE substr(number,1,2) = '{d}{d2}' OR
                      substr(number,2,2) = '{d}{d2}' OR
                      substr(number,3,2) = '{d}{d2}' OR
                      substr(number,4,2) = '{d}{d2}';
            """, ())[0]['prob_adjacent']
            print(f"{d}{d2}: {prob_adj*100:.4f} %")
    
    def probability_of_adjacent_digits_any_order(): 
        print(f"\nProbability that X and Y are adjacent in any order (XY or YX)")
        for d in range(0, 10):
            for d2 in range(0, 10):
                prob_adj_any = config.db.execute_custom_query(f"""
                    SELECT
                        COUNT(*) * 1.0 / (SELECT COUNT(*) FROM lottery) AS prob_adjacent
                    FROM lottery
                    WHERE substr(number,1,2) IN ('{d}{d2}', '{d2}{d}') OR
                        substr(number,2,2) IN ('{d}{d2}', '{d2}{d}') OR
                        substr(number,3,2) IN ('{d}{d2}', '{d2}{d}') OR
                        substr(number,4,2) IN ('{d}{d2}', '{d2}{d}');
                """, ())[0]['prob_adjacent']
                print(f"{d}{d2} - {d2}{d}: {prob_adj_any*100:.4f} %")
    
    def conditional_probability_xy_yx_x():
        print(f"\nConditional probability: P(XY or YX | X appears)")
        for d in range(0, 10):
            for d2 in range(0, 10):
                prob_cond = config.db.execute_custom_query(f"""
                    WITH any_X AS (
                        SELECT COUNT(*) AS cnt FROM lottery WHERE instr(number, '{d}') > 0
                    ),
                    X_has_Y AS (
                        SELECT COUNT(*) AS cnt FROM lottery
                        WHERE substr(number,1,2) IN ('{d}{d2}', '{d2}{d}') OR
                            substr(number,2,2) IN ('{d}{d2}', '{d2}{d}') OR
                            substr(number,3,2) IN ('{d}{d2}', '{d2}{d}') OR
                            substr(number,4,2) IN ('{d}{d2}', '{d2}{d}')
                    )
                    SELECT
                        any_X.cnt AS count_any_X,
                        X_has_Y.cnt AS count_X_has_Y,
                        CASE WHEN any_X.cnt = 0 THEN 0 ELSE X_has_Y.cnt * 1.0 / any_X.cnt END AS prob_conditional
                    FROM any_X, X_has_Y;
                """, ())[0]['prob_conditional']
                print(f"{d}{d2}: {prob_cond*100:.4f} %")

def calculate_probabilities_ia(type="all"):
    def predict_and_save(sequences, type):
        print(f"\n[*]  Predicting numbers for {type}...")
        sequences = [list(map(int, sequence.get("number"))) for sequence in sequences]
        # OUTPUT: [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], ...]

        new_predictor = NumberPredictor()
        new_predictor.load(f"models/lottery_number_predictor_{type}.pth")
        print("\n[*] LOTERIA ")
        for i in range(1, 6):
            print(''.join(str(k) for k in new_predictor.predict_next(sequences[-i], top_k=5).keys()))
        print("\n[*] SERIES")
        print('\n'.join([serie for serie in list(new_predictor.count_series(series).to_dict().keys())[:5]]))
        helpers.save_result_to_json(data={
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": type,
            "last_sequence": ''.join(str(n) for n in sequences[-1]),
            "last_series": list(new_predictor.count_series(series).to_dict().keys())[-1],
            "lottery": [str(k) for i in range(1, 6) for k in new_predictor.predict_next(sequences[-i], top_k=5).keys()],
            "series": list(new_predictor.count_series(series).to_dict().keys())[:5]
        }, filename="data/last_numbers.json")
        print(f"\n[?]  Results saved to JSON\n")
        
        # get sorted numbers from DB
    sequences = sorted(config.db.get_all("lottery"), 
                    key=lambda x: (x["year"], x["month"]))
    series = [sequence.get("series") for sequence in sequences]
    
    
    #TODO: Calculate individual probabilities by game
    types_all=config.commands2
    types_all.remove("all")
    if type != "all":
        if not os.path.exists(f"models/lottery_number_predictor_{type}.pth"):
            print("[!] No trained model found. Please train the model first.")
            return
        sequences_filtered = [s for s in sequences if type.lower() in s.get("type").lower()]
        sequences_filtered = sorted(sequences_filtered, key=lambda x: (x["year"], x["month"]))
        print(f"[*]  Calculating probabilities for {type}...\n")
        predict_and_save(sequences_filtered, type)
    else:
        print(f"[*]  Calculating probabilities for all types...\n")
        for t in types_all:
            if not os.path.exists(f"models/lottery_number_predictor_{t}.pth"):
                print(f"[!]  No trained model found for type {t}. Please train the model first.")
                continue
            print(f"[*]  Calculating probabilities for {t}...\n")
            sequences_filtered = [s for s in sequences if t.lower() in s.get("type").lower()]
            sequences_filtered = sorted(sequences_filtered, key=lambda x: (x["year"], x["month"]))
            predict_and_save(sequences_filtered, t)

def train_ia(type="all"):
    def train_and_save_model(sequences, t):
        print(f"\n[*]  Training model on {len(sequences)} sequences from type {t}...")
        print(f"\n[?]  Last number registered: \n>Number: {sequences[-1].get('number')}\n>Series: {sequences[-1].get('series')}\n>Date: {sequences[-1].get('month')}/{sequences[-1].get('year')}")
        sequences = [list(map(int, sequence.get("number"))) for sequence in sequences]
        # OUTPUT: [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [...]]
        
        # train
        predictor = NumberPredictor(epochs=160)
        predictor.train(sequences)

        # predict
        input_seq = sequences[-1]  # last sequence as input
        probs = predictor.predict_next(input_seq)
        print("[*]  Probabilities:", probs)

        name=f"models/lottery_number_predictor_{t}.pth"
        
        # save and load
        predictor.save(name)
        print(f"\n[*]  Model saved as {name}\n")

        print(f"\n[*]  Model {name} trained successfully.")
    
    print(f"\n[*]  Starting training for {type}... ")
    # sort list to make a meaningful sequence sense while training
    sequences = sorted(config.db.get_all("lottery"), 
                       key=lambda x: (x["year"], x["month"]))
    if not sequences:
        print("[!]  No sequences found.")
        return
    if type != "all":
        sequences_filtered = [s for s in sequences if type.lower() in s.get("type").lower()]
        if not sequences_filtered:
            print(f"[!]  No sequences found by name {type}.")
            return
        sequences_filtered = sorted(sequences_filtered, key=lambda x: (x["year"], x["month"]))
        if type not in config.commands2:
            print(f"[!]  No type found for {type}")
        train_and_save_model(sequences_filtered, type)
        
    else:
        types_all = config.commands2
        types_all.remove("all")
        for t in types_all:
            sequences_filtered = [s for s in sequences if t.lower() in s.get("type").lower()]
            sequences_filtered = sorted(sequences_filtered, key=lambda x: (x["year"], x["month"]))
            if not sequences_filtered:
                print(f"[!]  No sequences found by name {t}.")
                return
            train_and_save_model(sequences_filtered, t)
            
if __name__ == "__main__":
    print("##############################################")
    print(config.ascii_art)
    while True:
        print("##############################################")
        print("help\t\t-\tShow this help message")
        print("train\t\t-\tTrain the model")
        print("populate\t-\tPopulate the database")
        print("update\t\t-\tUpdate the database (since 01/01 from current year)")
        print("predict\t\t-\tPredict the most probable result")
        print("exit\t\t-\tExit the program")
        print("##############################################")
        match config.prompt("$> ", 
                            completer=config.commands_completer).strip().lower():
            
            case "update":
                update_db()
            case "populate":
                populate_db()
            case "train":
                print("Select type to train:")
                print("all", "cuponazo", "diario", "madre", "verano", "navidad", "padre", "sueldazo", sep=" | ")
                match config.prompt("$> ", completer=config.commands_completer2).strip().lower():
                    case "all":
                        train_ia()
                    case "cuponazo":
                        train_ia("cuponazo")
                    case "diario":
                        train_ia("diario")
                    case "madre":
                        train_ia("madre")
                    case "verano":
                        train_ia("verano")
                    case "navidad":
                        train_ia("navidad")
                    case "padre":
                        train_ia("padre")
                    case "sueldazo":
                        train_ia("sueldazo")
            case "predict":
                print("Select type to predict:")
                print("all", "cuponazo", "diario", "madre", "verano", "navidad", "padre", "sueldazo", sep=" | ")
                match config.prompt("$> ", completer=config.commands_completer2).strip().lower():
                    case "all":
                        calculate_probabilities_ia()
                    case "cuponazo":
                        calculate_probabilities_ia("cuponazo")
                    case "diario":
                        calculate_probabilities_ia("diario")
                    case "madre":
                        calculate_probabilities_ia("madre")
                    case "verano":
                        calculate_probabilities_ia("verano")
                    case "navidad":
                        calculate_probabilities_ia("navidad")
                    case "padre":
                        calculate_probabilities_ia("padre")
                    case "sueldazo":
                        calculate_probabilities_ia("sueldazo")
            case "exit":
                break
