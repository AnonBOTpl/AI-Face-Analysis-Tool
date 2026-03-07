import unittest
from unittest.mock import patch, MagicMock, call
import tkinter as tk
import face_gui

class TestFaceGui(unittest.TestCase):
    @patch('face_gui.tk')
    @patch('face_gui.ttk')
    def test_faceapp_initialization(self, mock_ttk, mock_tk):
        mock_root = MagicMock()

        # Mocking variables properly
        mock_string_var = MagicMock()
        mock_string_var.get.return_value = 'opencv'
        mock_tk.StringVar.return_value = mock_string_var

        app = face_gui.FaceApp(mock_root)

        self.assertEqual(app.detector_var.get(), 'opencv')
        self.assertIsNone(app.img_path)
        mock_root.title.assert_called_once_with("DeepFace Face Detector")

    @patch('face_gui.tk')
    @patch('face_gui.ttk')
    @patch('face_gui.filedialog.askopenfilename')
    def test_select_file_success(self, mock_askopenfilename, mock_ttk, mock_tk):
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        app = face_gui.FaceApp(mock_root)

        mock_askopenfilename.return_value = '/path/to/image.jpg'
        app.select_file()

        self.assertEqual(app.img_path, '/path/to/image.jpg')

    @patch('face_gui.tk')
    @patch('face_gui.ttk')
    @patch('face_gui.filedialog.askopenfilename')
    def test_select_file_cancel(self, mock_askopenfilename, mock_ttk, mock_tk):
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        app = face_gui.FaceApp(mock_root)
        app.img_path = '/old/path.jpg'

        mock_askopenfilename.return_value = ''
        app.select_file()

        self.assertIsNone(app.img_path)

    @patch('face_gui.tk')
    @patch('face_gui.ttk')
    @patch('face_gui.messagebox.showerror')
    def test_analyze_no_file(self, mock_showerror, mock_ttk, mock_tk):
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        app = face_gui.FaceApp(mock_root)
        app.img_path = None

        app.analyze()

        mock_showerror.assert_called_once()
        self.assertTrue("Nie wybrano pliku" in str(mock_showerror.call_args) or "No file selected" in str(mock_showerror.call_args))

    @patch('face_gui.tk')
    @patch('face_gui.ttk')
    @patch('face_gui.messagebox.showerror')
    def test_analyze_no_actions(self, mock_showerror, mock_ttk, mock_tk):
        mock_root = MagicMock()

        # For mock_tk.BooleanVar, we need it to return False when get() is called
        # because the test depends on the values being False.
        # But app initialized them with mock_tk.BooleanVar().
        # Let's manually set the get() return value on the existing variables.

        app = face_gui.FaceApp(mock_root)
        app.img_path = 'image.jpg'

        # Deselect all actions
        for var in app.action_vars.values():
            var.get.return_value = False

        app.analyze()

        mock_showerror.assert_called_once()
        self.assertTrue("przynajmniej jedną akcję" in str(mock_showerror.call_args))

    @patch('face_gui.tk')
    @patch('face_gui.ttk')
    @patch('face_gui.threading.Thread')
    def test_analyze_starts_thread(self, mock_thread, mock_ttk, mock_tk):
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        app = face_gui.FaceApp(mock_root)
        app.img_path = 'image.jpg'

        # Select all actions
        for var in app.action_vars.values():
            var.set(True)

        app.analyze()

        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @patch('face_gui.tk')
    @patch('face_gui.ttk')
    @patch('face_gui.Image.open')
    @patch('face_gui.tempfile.mkstemp')
    @patch('face_gui.os.close')
    @patch('shutil.copy2')
    @patch('face_gui.messagebox.showerror')
    def test_analyze_worker_image_copy_fail(self, mock_showerror, mock_copy2, mock_close, mock_mkstemp, mock_image_open, mock_ttk, mock_tk):
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        app = face_gui.FaceApp(mock_root)
        app.img_path = 'image.jpg'

        mock_mkstemp.return_value = (1, 'temp.jpg')
        mock_copy2.side_effect = Exception("Copy failed")

        app._analyze_worker('opencv', ['age'])

        # In tk we use master.after to schedule GUI updates, so we need to manually call the lambda
        mock_root.after.assert_called()
        # Find the lambda for showerror and call it
        for call_args in mock_root.after.call_args_list:
            if call_args[0][0] == 0:
                func = call_args[0][1]
                func()

        mock_showerror.assert_called()

    @patch('face_gui.tk')
    @patch('face_gui.ttk')
    @patch('face_gui.tempfile.mkstemp')
    @patch('face_gui.os.close')
    @patch('shutil.copy2')
    @patch('face_gui.DeepFace.analyze')
    @patch('face_gui.messagebox.showerror')
    def test_analyze_worker_deepface_fail(self, mock_showerror, mock_analyze, mock_copy2, mock_close, mock_mkstemp, mock_ttk, mock_tk):
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        app = face_gui.FaceApp(mock_root)
        app.img_path = 'image.jpg'

        mock_mkstemp.return_value = (1, 'temp.jpg')
        mock_analyze.side_effect = Exception("Deepface fail")

        app._analyze_worker('opencv', ['age'])

        mock_root.after.assert_called()
        for call_args in mock_root.after.call_args_list:
            if call_args[0][0] == 0:
                func = call_args[0][1]
                func()

        mock_showerror.assert_called()
        self.assertTrue("Wystąpił błąd" in str(mock_showerror.call_args))

if __name__ == '__main__':
    unittest.main()
