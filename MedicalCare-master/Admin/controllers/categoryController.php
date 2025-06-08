<?php
require_once('./models/category.php');
class CategoryController
{
  var $category_model;
  public function __construct()
  {
    $this->category_model = new Category();
  }
  public function list()
  {
    $categories = $this->category_model->getCategories();
    require_once('./Views/index.php');
  }
  public function edit()
  {
    $id = filter_input(INPUT_GET, 'id', FILTER_VALIDATE_INT);
    $category = $this->category_model->getCategory($id);
    require_once('./views/index.php');
  }
  public function update()
  {
    $id = $_POST['id'];
    $name = $_POST['name'];
    if (isset($_FILES['image']) && $_FILES['image']['error'] === UPLOAD_ERR_OK && isset($_FILES['icon']) && $_FILES['icon']['error'] === UPLOAD_ERR_OK) {
      // Get the image name and path
      $image_path = basename($_FILES['image']['name']);
      $target_image_dir = '../public/img/images/category/';
      $target_image_file = $target_image_dir . $image_path;
      $imageFileType = strtolower(pathinfo($target_image_file, PATHINFO_EXTENSION));

      // Get the icon name and path
      $icon_path = basename($_FILES['icon']['name']);
      $target_icon_dir = '../public/img/images/icons/';
      $target_icon_file = $target_icon_dir . $icon_path;
      $iconFileType = strtolower(pathinfo($target_icon_file, PATHINFO_EXTENSION));
      // Check if the file type is allowed
      $allowedFormats = array("jpg", "jpeg", "png");
      if (!in_array($imageFileType, $allowedFormats) && !in_array($iconFileType, $allowedFormats)) {
        $error = 'Only JPG, JPEG, and PNG files are allowed.';
      } else {
        // Use move_uploaded_file correctly with the temporary file name
        if (move_uploaded_file($_FILES['image']['tmp_name'], $target_image_file) && move_uploaded_file($_FILES['icon']['tmp_name'], $target_icon_file)) {
          $this->category_model->update($id, $name, $icon_path, $image_path);
        } else {
          header('Location: ?act=error');
        }
      }
    } else {
      header('Location: ?act=error');
    }
  }
  public function delete()
  {
    $id = $_GET['id'];
    $this->category_model->delete($id);
  }
  public function add() {
    $this->category_model->add();
  }
  public function store()
  {
    $name = $_POST['name'];
    if (isset($_FILES['image']) && $_FILES['image']['error'] === UPLOAD_ERR_OK && isset($_FILES['icon']) && $_FILES['icon']['error'] === UPLOAD_ERR_OK) {
      // Get the image name and path
      $image_path = basename($_FILES['image']['name']);
      $target_image_dir = '../public/img/images/category/';
      $target_image_file = $target_image_dir . $image_path;
      $imageFileType = strtolower(pathinfo($target_image_file, PATHINFO_EXTENSION));

      // Get the icon name and path
      $icon_path = basename($_FILES['icon']['name']);
      $target_icon_dir = '../public/img/images/icons/';
      $target_icon_file = $target_icon_dir . $icon_path;
      $iconFileType = strtolower(pathinfo($target_icon_file, PATHINFO_EXTENSION));
      // Check if the file type is allowed
      $allowedFormats = array("jpg", "jpeg", "png");
      if (!in_array($imageFileType, $allowedFormats) && !in_array($iconFileType, $allowedFormats)) {
        header('Location: ?act=error');
      } else {
        // Use move_uploaded_file correctly with the temporary file name
        if (move_uploaded_file($_FILES['image']['tmp_name'], $target_image_file) && move_uploaded_file($_FILES['icon']['tmp_name'], $target_icon_file)) {
          $this->category_model->store($name, $icon_path, $image_path);
        } else {
          header('Location: ?act=error');
        }
      }
    }
  }
}
