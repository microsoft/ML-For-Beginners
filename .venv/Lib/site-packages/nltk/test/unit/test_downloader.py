from nltk import download


def test_downloader_using_existing_parent_download_dir(tmp_path):
    """Test that download works properly when the parent folder of the download_dir exists"""

    download_dir = str(tmp_path.joinpath("another_dir"))
    download_status = download("mwa_ppdb", download_dir)
    assert download_status is True


def test_downloader_using_non_existing_parent_download_dir(tmp_path):
    """Test that download works properly when the parent folder of the download_dir does not exist"""

    download_dir = str(
        tmp_path.joinpath("non-existing-parent-folder", "another-non-existing-folder")
    )
    download_status = download("mwa_ppdb", download_dir)
    assert download_status is True
