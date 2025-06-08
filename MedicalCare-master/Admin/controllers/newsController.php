<?php
require_once('./models/news.php');
class NewsController
{
  var $news_model;
  public function __construct()
  {
    $this->news_model = new News();
  }
  public function list()
  {
    $news = $this->news_model->getNews();
    require_once('./views/index.php');
  }
  public function add()
  {
    require_once('./Views/index.php');
  }
  public function edit()
  {
    $id = filter_input(INPUT_GET, 'id', FILTER_VALIDATE_INT);
    $new = $this->news_model->getNew($id);
    require_once('./views/index.php');
  }
  public function update()
  {
      $id = $_POST['id'];
      $title = $_POST['title'];
      $content = $_POST['content'];
  
      // Check if the image file is provided
      if (isset($_FILES['image']) && $_FILES['image']['error'] === UPLOAD_ERR_OK) {
          // Get the file name and path
          $image_path = basename($_FILES['image']['name']);
          $target_dir = '../public/img/images/news/';
          $target_file = $target_dir . $image_path;
          $imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));
  
          // Check if the file type is allowed
          $allowedFormats = array("jpg", "jpeg", "png");
          if (!in_array($imageFileType, $allowedFormats)) {
              $error = 'Only JPG, JPEG, and PNG files are allowed.';
          } else {
              // Use move_uploaded_file correctly with the temporary file name
              if (move_uploaded_file($_FILES['image']['tmp_name'], $target_file)) {
                  // Call the update function in the news model
                  $this->news_model->update($id, $title, $content, $image_path);
              } else {
                  header('Location: ?act=error');
              }
          }
      } else {
          // If no image file is provided, update without changing the image
          $this->news_model->updateWithoutImage($id, $title, $content);
      }
  
      // Redirect to the appropriate location (adjust the location as needed)
      header('Location: ?act=news');
  }
  
  public function delete()
  {
    $id = $_GET['id'];
    $this->news_model->delete($id);
  }
  public function store()
  {
    $title = $_POST['title'];
    $content = $_POST['content'];
    if (isset($_FILES['image']) && $_FILES['image']['error'] === UPLOAD_ERR_OK) {
      // Get the file name and path
      $image_path = basename($_FILES['image']['name']);
      $target_dir = '../public/img/images/news/';
      $target_file = $target_dir . $image_path;
      $imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));

      // Check if the file type is allowed
      $allowedFormats = array("jpg", "jpeg", "png");
      if (!in_array($imageFileType, $allowedFormats)) {
        $error = 'Only JPG, JPEG, and PNG files are allowed.';
      } else {
        // Use move_uploaded_file correctly with the temporary file name
        if (move_uploaded_file($_FILES['image']['tmp_name'], $target_file)) {
          $this->news_model->store($title, $content, $image_path);
        } else {
          header('Location: ?act=error');
        }
      }
    } else {
      header('Location: ?act=error');
    }
  }
}
