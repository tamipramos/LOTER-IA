import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
import tempfile
from io import StringIO

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lottery import Lottery
from helpers import parse_result, split_results, save_result_to_json
from dbcontroller import SQLiteController
from loteria import NumberPredictor, NumberSequenceDataset
import torch


class TestLottery(unittest.TestCase):
    """Test cases for Lottery class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_data = {
            "type": "Cuponazo",
            "number": "12345",
            "series": "001",
            "year": 2025,
            "month": 1
        }

    def test_lottery_creation(self):
        """Test Lottery object creation"""
        lottery = Lottery(**self.test_data)
        self.assertEqual(lottery.type, "Cuponazo")
        self.assertEqual(lottery.number, "12345")
        self.assertEqual(lottery.series, "001")
        self.assertEqual(lottery.year, 2025)
        self.assertEqual(lottery.month, 1)

    def test_lottery_to_dict(self):
        """Test Lottery to_dict conversion"""
        lottery = Lottery(**self.test_data)
        result = lottery.to_dict()
        self.assertEqual(result, self.test_data)

    def test_lottery_from_dict(self):
        """Test Lottery from_dict creation"""
        lottery = Lottery.from_dict(self.test_data)
        self.assertEqual(lottery.type, self.test_data["type"])
        self.assertEqual(lottery.number, self.test_data["number"])

    def test_lottery_from_dict_missing_fields(self):
        """Test Lottery from_dict with missing fields"""
        incomplete_data = {"type": "Cuponazo", "number": "12345"}
        lottery = Lottery.from_dict(incomplete_data)
        self.assertEqual(lottery.series, None)
        self.assertEqual(lottery.year, None)

    def test_lottery_exists_in_db(self):
        """Test exists_in_db method"""
        lottery = Lottery(**self.test_data)
        mock_db = Mock()
        mock_db.get_by_id_custom.return_value = None
        
        result = lottery.exists_in_db(mock_db)
        self.assertFalse(result)
        mock_db.get_by_id_custom.assert_called_once()

    def test_lottery_save_to_db_new(self):
        """Test save_to_db for new entry"""
        lottery = Lottery(**self.test_data)
        mock_db = Mock()
        mock_db.get_by_id_custom.return_value = None
        
        lottery.save_to_db(mock_db)
        mock_db.insert.assert_called_once()

    def test_lottery_save_to_db_existing(self):
        """Test save_to_db for existing entry (should skip)"""
        lottery = Lottery(**self.test_data)
        mock_db = Mock()
        mock_db.get_by_id_custom.return_value = {"id": 1}
        
        lottery.save_to_db(mock_db)
        mock_db.insert.assert_not_called()


class TestHelpers(unittest.TestCase):
    """Test cases for helpers module"""

    def test_parse_result_complete(self):
        """Test parse_result with complete data"""
        text = """Cuponazo
Número
12345
Serie
001"""
        result = parse_result(text)
        self.assertEqual(result["type"], "Cuponazo")
        self.assertEqual(result["number"], "12345")
        self.assertEqual(result["series"], "001")

    def test_parse_result_missing_serie(self):
        """Test parse_result without serie"""
        text = """Cupón Diario
Número
54321"""
        result = parse_result(text)
        self.assertEqual(result["type"], "Cupón Diario")
        self.assertEqual(result["number"], "54321")
        self.assertIsNone(result["series"])

    def test_parse_result_empty(self):
        """Test parse_result with empty text"""
        text = ""
        result = parse_result(text)
        self.assertIsNone(result["type"])

    def test_split_results_single(self):
        """Test split_results with single result"""
        text = """Cuponazo
Número
12345
Serie
001"""
        blocks = split_results(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("Cuponazo", blocks[0])

    def test_split_results_multiple(self):
        """Test split_results with multiple results"""
        text = """Cuponazo
Número
12345
Serie
001

Cupón Diario
Número
54321
Serie
002"""
        blocks = split_results(text)
        self.assertEqual(len(blocks), 2)

    def test_split_results_with_bono(self):
        """Test split_results recognizes Bono variant"""
        text = """Cuponazo
Número
12345

Bono Cupoón
Número
99999"""
        blocks = split_results(text)
        self.assertEqual(len(blocks), 2)

    def test_save_result_to_json_new_file(self):
        """Test save_result_to_json creates new file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        os.remove(temp_file)
        
        data = {"type": "Cuponazo", "number": "12345"}
        save_result_to_json(data, temp_file)
        
        self.assertTrue(os.path.exists(temp_file))
        with open(temp_file, 'r') as f:
            content = json.load(f)
            self.assertEqual(len(content), 1)
            self.assertEqual(content[0], data)
        
        os.remove(temp_file)

    def test_save_result_to_json_append(self):
        """Test save_result_to_json appends to existing file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            json.dump([{"type": "Existing"}], f)
        
        data = {"type": "Cuponazo", "number": "12345"}
        save_result_to_json(data, temp_file)
        
        with open(temp_file, 'r') as f:
            content = json.load(f)
            self.assertEqual(len(content), 2)
            self.assertEqual(content[1], data)
        
        os.remove(temp_file)


class TestSQLiteController(unittest.TestCase):
    """Test cases for SQLiteController"""

    def setUp(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = SQLiteController(self.temp_db.name)

    def tearDown(self):
        """Clean up test database"""
        self.db.close()
        os.remove(self.temp_db.name)

    def test_create_table(self):
        """Test table creation"""
        columns = {
            "id": "INTEGER PRIMARY KEY",
            "name": "TEXT NOT NULL"
        }
        self.db.create_table("test_table", columns)
        
        cursor = self.db.cursor
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)

    def test_insert_and_get_all(self):
        """Test insert and retrieve data"""
        columns = {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}
        self.db.create_table("test_table", columns)
        
        data = {"name": "Test"}
        self.db.insert("test_table", data)
        
        results = self.db.get_all("test_table")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Test")

    def test_get_by_id(self):
        """Test get_by_id method"""
        columns = {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}
        self.db.create_table("test_table", columns)
        
        self.db.insert("test_table", {"name": "Test1"})
        self.db.insert("test_table", {"name": "Test2"})
        
        result = self.db.get_by_id("test_table", 1)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test1")

    def test_delete_by_id(self):
        """Test delete_by_id method"""
        columns = {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}
        self.db.create_table("test_table", columns)
        
        self.db.insert("test_table", {"name": "Test"})
        self.db.delete_by_id("test_table", 1)
        
        result = self.db.get_by_id("test_table", 1)
        self.assertIsNone(result)

    def test_execute_custom_query(self):
        """Test execute_custom_query method"""
        columns = {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}
        self.db.create_table("test_table", columns)
        
        self.db.insert("test_table", {"name": "Test1"})
        self.db.insert("test_table", {"name": "Test2"})
        
        query = "SELECT * FROM test_table WHERE name = ?"
        results = self.db.execute_custom_query(query, ("Test1",))
        self.assertEqual(len(results), 1)


class TestNumberPredictor(unittest.TestCase):
    """Test cases for NumberPredictor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.sequences = [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]
        ]

    def test_number_predictor_init(self):
        """Test NumberPredictor initialization"""
        predictor = NumberPredictor(embed_dim=32, hidden_dim=64, lr=0.001)
        self.assertEqual(predictor.embed_dim, 32)
        self.assertEqual(predictor.hidden_dim, 64)
        self.assertIsNone(predictor.model)

    def test_prepare_vocab(self):
        """Test prepare_vocab method"""
        predictor = NumberPredictor()
        predictor.prepare_vocab(self.sequences)
        
        self.assertEqual(predictor.vocab_size, 7)  # Numbers 1-7
        self.assertIn(1, predictor.number_to_id)
        self.assertIn(0, predictor.id_to_number)

    def test_number_sequence_dataset(self):
        """Test NumberSequenceDataset"""
        number_to_id = {i: i for i in range(10)}
        dataset = NumberSequenceDataset(self.sequences, number_to_id)
        
        self.assertEqual(len(dataset), 3)
        
        input_seq, target_seq = dataset[0]
        self.assertEqual(len(input_seq), 4)  # seq[:-1]
        self.assertEqual(len(target_seq), 4)  # seq[1:]

    def test_train_model(self):
        """Test model training"""
        predictor = NumberPredictor(epochs=2, batch_size=2)
        predictor.train(self.sequences)
        
        self.assertIsNotNone(predictor.model)
        self.assertEqual(predictor.vocab_size, 7)

    def test_predict_next(self):
        """Test predict_next method"""
        predictor = NumberPredictor(epochs=2, batch_size=2)
        predictor.train(self.sequences)
        
        predictions = predictor.predict_next([1, 2, 3, 4], top_k=3)
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(isinstance(v, float) for v in predictions.values()))

    def test_save_and_load_model(self):
        """Test save and load model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pth")
            
            predictor1 = NumberPredictor(epochs=2)
            predictor1.train(self.sequences)
            predictor1.save(model_path)
            
            predictor2 = NumberPredictor()
            predictor2.load(model_path)
            
            self.assertEqual(predictor1.vocab_size, predictor2.vocab_size)
            self.assertEqual(predictor1.number_to_id, predictor2.number_to_id)

    def test_sequence_log_probability(self):
        """Test sequence_log_probability method"""
        predictor = NumberPredictor(epochs=2)
        predictor.train(self.sequences)
        
        log_prob = predictor.sequence_log_probability([1, 2, 3, 4, 5])
        # Accept numpy float32 and Python float/int
        self.assertTrue(isinstance(log_prob, (int, float)) or hasattr(log_prob, 'item'))
        self.assertLessEqual(float(log_prob), 0)  # Log probabilities are negative

    def test_generate_sequences(self):
        """Test generate_sequences method"""
        predictor = NumberPredictor(epochs=2, batch_size=2)
        predictor.train(self.sequences)
        
        generated = predictor.generate_sequences([1, 2, 3], length=5, top_k=2)
        self.assertEqual(len(generated), 2)
        
        for seq, log_prob in generated:
            self.assertEqual(len(seq), 5)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def setUp(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = SQLiteController(self.temp_db.name)
        self.db.create_table("lottery", {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "type": "TEXT NOT NULL",
            "number": "TEXT",
            "series": "TEXT",
            "year": "INTEGER",
            "month": "INTEGER"
        })

    def tearDown(self):
        """Clean up"""
        self.db.close()
        os.remove(self.temp_db.name)

    def test_lottery_workflow(self):
        """Test complete workflow: create, parse, save"""
        raw_text = """Cuponazo
Número
12345
Serie
001"""
        
        parsed = parse_result(raw_text)
        parsed["year"] = 2025
        parsed["month"] = 1
        
        lottery = Lottery.from_dict(parsed)
        self.assertFalse(lottery.exists_in_db(self.db))
        
        lottery.save_to_db(self.db)
        self.assertTrue(lottery.exists_in_db(self.db))


if __name__ == "__main__":
    unittest.main()
