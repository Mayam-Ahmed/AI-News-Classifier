import unittest
import joblib

model = joblib.load("news_model.pkl")

class FakeNewsTest(unittest.TestCase):

    def test_real_news(self):
        text = "Government launches new education reform policy"
        prediction = model.predict([text])[0]
        self.assertEqual(prediction, "REAL")

    def test_fake_news(self):
        text = "Aliens officially confirmed in Karachi government report"
        prediction = model.predict([text])[0]
        self.assertEqual(prediction, "FAKE")

    def test_edge_case(self):
        text = "Scientists discover new unknown energy source"
        prediction = model.predict([text])[0]
        self.assertIn(prediction, ["REAL", "FAKE"])

if __name__ == "__main__":
    unittest.main()