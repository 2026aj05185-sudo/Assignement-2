import io
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from evaluation import MODEL_FILES


ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_bundled_data_and_model_selection(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 6)
        self.assertEqual(app.metric[0].value, "0.5846")
        self.assertEqual(len(app.dataframe[0].value), 5)
        for name in MODEL_FILES:
            app.selectbox[0].select(name).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.metric), 6)
        app.toggle[0].set_value(True).run()
        self.assertEqual(len(app.exception), 0)

    def test_empty_upload_state(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.radio[0].set_value("Upload CSV").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.info[0].value, "No test CSV selected.")

    def test_invalid_upload_shows_error(self):
        uploaded = io.BytesIO(b"wrong,quality\n1,5\n")
        uploaded.name = "invalid.csv"
        with patch("streamlit.file_uploader", return_value=uploaded), \
            patch("streamlit.radio", return_value="Upload CSV"):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Missing required columns", app.error[0].value)

    def test_valid_uploaded_csv(self):
        uploaded = io.BytesIO((ROOT / "test_data.csv").read_bytes())
        uploaded.name = "uploaded_test.csv"
        with patch("streamlit.file_uploader", return_value=uploaded), \
                patch("streamlit.radio", return_value="Upload CSV"):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].value, "0.5846")

    def test_bundled_mode_ignores_previous_upload(self):
        uploaded = io.BytesIO(b"wrong,quality\n1,5\n")
        uploaded.name = "previous_upload.csv"
        with patch("streamlit.file_uploader", return_value=uploaded):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(app.metric[0].value, "0.5846")


if __name__ == "__main__":
    unittest.main()