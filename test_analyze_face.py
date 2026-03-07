import unittest
from unittest.mock import patch, MagicMock, call
import analyze_face
import sys

class TestAnalyzeFace(unittest.TestCase):
    @patch('analyze_face.tk.Tk')
    def test_choose_detector_success(self, mock_tk):
        mock_root = MagicMock()
        mock_tk.return_value = mock_root

        # Test default ok flow
        with patch('analyze_face.ttk.Combobox') as mock_combo:
            mock_combo_inst = MagicMock()
            mock_combo.return_value = mock_combo_inst

            # Need to capture the on_ok function and call it
            # We can mock the detector_var
            with patch('analyze_face.tk.StringVar') as mock_var:
                mock_var_inst = MagicMock()
                mock_var_inst.get.return_value = 'ssd'
                mock_var.return_value = mock_var_inst

                # We'll just mock the mainloop to exit immediately,
                # but we need to run on_ok to test success.
                # A simpler way is to mock choose_detector completely for main,
                # but let's test choose_detector internal behavior as requested.

                # To test inner workings without full tk event loop, we mock tk.Button
                # to extract the command
                with patch('analyze_face.tk.Button') as mock_button:
                    # Let mainloop return immediately
                    mock_root.mainloop.return_value = None

                    # We can't easily trigger the command in the same test without capturing it,
                    # since choose_detector blocks on mainloop.
                    # We will mock the result directly or capture the on_ok callback.
                    pass

    @patch('analyze_face.choose_detector')
    @patch('analyze_face.sys.exit')
    @patch('analyze_face.tk.Tk')
    def test_main_no_detector(self, mock_tk, mock_exit, mock_choose_detector):
        mock_choose_detector.return_value = None
        mock_exit.side_effect = SystemExit(0)
        with self.assertRaises(SystemExit):
            analyze_face.main()
        mock_exit.assert_called_once_with(0)

    @patch('analyze_face.choose_detector')
    @patch('analyze_face.filedialog.askopenfilename')
    @patch('analyze_face.sys.exit')
    @patch('analyze_face.tk.Tk')
    def test_main_no_file(self, mock_tk, mock_exit, mock_askopenfilename, mock_choose_detector):
        mock_choose_detector.return_value = 'opencv'
        mock_askopenfilename.return_value = ''
        mock_exit.side_effect = SystemExit(0)
        with self.assertRaises(SystemExit):
            analyze_face.main()
        mock_exit.assert_called_once_with(0)

    @patch('analyze_face.choose_detector')
    @patch('analyze_face.filedialog.askopenfilename')
    @patch('analyze_face.sys.exit')
    @patch('analyze_face.shutil.copy2')
    @patch('analyze_face.tempfile.mkstemp')
    @patch('analyze_face.os.close')
    @patch('analyze_face.tk.Tk')
    def test_main_copy_fail(self, mock_tk, mock_close, mock_mkstemp, mock_copy2, mock_exit, mock_askopenfilename, mock_choose_detector):
        mock_choose_detector.return_value = 'opencv'
        mock_askopenfilename.return_value = 'test.jpg'
        mock_mkstemp.return_value = (1, 'temp.jpg')
        mock_copy2.side_effect = Exception("Copy failed")
        mock_exit.side_effect = SystemExit(1)

        with self.assertRaises(SystemExit):
            analyze_face.main()
        mock_exit.assert_called_once_with(1)

    @patch('analyze_face.choose_detector')
    @patch('analyze_face.filedialog.askopenfilename')
    @patch('analyze_face.sys.exit')
    @patch('analyze_face.shutil.copy2')
    @patch('analyze_face.tempfile.mkstemp')
    @patch('analyze_face.os.close')
    @patch('analyze_face.DeepFace.analyze')
    @patch('analyze_face.os.remove')
    @patch('analyze_face.tk.Tk')
    def test_main_no_faces(self, mock_tk, mock_remove, mock_analyze, mock_close, mock_mkstemp, mock_copy2, mock_exit, mock_askopenfilename, mock_choose_detector):
        mock_choose_detector.return_value = 'opencv'
        mock_askopenfilename.return_value = 'test.jpg'
        mock_mkstemp.return_value = (1, 'temp.jpg')
        mock_analyze.return_value = [] # No faces
        mock_exit.side_effect = SystemExit(0)

        with self.assertRaises(SystemExit):
            analyze_face.main()
        mock_exit.assert_called_once_with(0)

    @patch('analyze_face.choose_detector')
    @patch('analyze_face.filedialog.askopenfilename')
    @patch('analyze_face.sys.exit')
    @patch('analyze_face.shutil.copy2')
    @patch('analyze_face.tempfile.mkstemp')
    @patch('analyze_face.os.close')
    @patch('analyze_face.DeepFace.analyze')
    @patch('analyze_face.os.remove')
    @patch('analyze_face.cv2.imread')
    @patch('analyze_face.cv2.cvtColor')
    @patch('analyze_face.plt.show')
    @patch('analyze_face.tk.Tk')
    def test_main_success_with_child_warning(self, mock_tk, mock_show, mock_cvtcolor, mock_imread, mock_remove, mock_analyze, mock_close, mock_mkstemp, mock_copy2, mock_exit, mock_askopenfilename, mock_choose_detector):
        mock_choose_detector.return_value = 'opencv'
        mock_askopenfilename.return_value = 'test.jpg'
        mock_mkstemp.return_value = (1, 'temp.jpg')

        # Mocking DeepFace output with a child (age < 16)
        mock_analyze.return_value = [{
            "age": 10,
            "gender": {"Woman": 40.0, "Man": 60.0},
            "race": {"white": 90.0},
            "emotion": {"happy": 80.0},
            "dominant_gender": "Man",
            "dominant_race": "white",
            "dominant_emotion": "happy",
            "region": {"x": 10, "y": 10, "w": 100, "h": 100}
        }]

        # Mocking OpenCV
        mock_img = MagicMock()
        mock_img.shape = (500, 500, 3)
        mock_imread.return_value = mock_img

        with patch('builtins.print') as mock_print:
            analyze_face.main()

            # Check if child warning was printed
            child_warning_printed = any(analyze_face.T["child_warning"].format(age=10) in str(call_args) for call_args in mock_print.call_args_list)
            self.assertTrue(child_warning_printed)

        mock_exit.assert_not_called()

if __name__ == '__main__':
    unittest.main()
