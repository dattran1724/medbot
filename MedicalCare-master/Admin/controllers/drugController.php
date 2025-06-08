<?php
require_once('./models/drug.php');
require_once('./models/category.php');
class DrugController
{
  var $drug_model;
  var $category_model;
  public function __construct()
  {
    $this->drug_model = new Drug();
    $this->category_model = new Category();
  }
  public function list()
  {
    $cat_id = filter_input(INPUT_POST, 'cat_id', FILTER_VALIDATE_INT);
    $categories = $this->category_model->getCategoryNames();
    if (isset($cat_id)) {
      $drugs = $this->drug_model->getDrugs($cat_id);
    } else {
      $drugs = $this->drug_model->getDrugs();
    }
    require_once('./views/index.php');
  }
  public function edit()
  {
    $id = filter_input(INPUT_GET, 'id', FILTER_VALIDATE_INT);
    $categories = $this->category_model->getCategories();
    $drug = $this->drug_model->getDrug($id);
    require_once('./views/index.php');
  }
  public function update()
  {
    $id = $_POST['id'];
    $name = $_POST['name'];
    $title_description = $_POST['description'];
    $category = $_POST['category_id'];
    $information = $_POST['information'];
    $instruction = $_POST['instruction'];
    $ingredient = $_POST['ingredient'];
    $attention = $_POST['attention'];
    $price = $_POST['price'];
    $stock = $_POST['quantity'];

    // Check if the image file is provided
    if (isset($_FILES['image']) && $_FILES['image']['error'] === UPLOAD_ERR_OK) {
      // Get the file name and path
      $image_path = basename($_FILES['image']['name']);
      $target_dir = '../public/img/images/drug/';
      $target_file = $target_dir . $image_path;
      $imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));

      // Check if the file type is allowed
      $allowedFormats = array("jpg", "jpeg", "png");
      if (!in_array($imageFileType, $allowedFormats)) {
        $error = 'Only JPG, JPEG, and PNG files are allowed.';
      } else {
        // Use move_uploaded_file correctly with the temporary file name
        if (move_uploaded_file($_FILES['image']['tmp_name'], $target_file)) {
          // Call the update function in the drug model
          $this->drug_model->update($id, $name, $title_description, $category, $information, $ingredient, $instruction, $attention, $price, $stock, $image_path);
        } else {
          header('Location: ?act=error');
        }
      }
    } else {
      // If no image file is provided, update without changing the image
      $this->drug_model->updateWithoutImage($id, $name, $title_description, $category, $information, $ingredient, $instruction, $attention, $price, $stock);
    }

    // Redirect to the appropriate location (adjust the location as needed)
    header('Location: ?act=drug');
  }

  public function delete()
  {
    $id = $_GET['id'];
    $this->drug_model->delete($id);
  }
  public function add()
  {
    $categories = $this->category_model->getCategoryNames();
    require_once('./Views/index.php');
  }
  public function store()
  {
    $name = $_POST['name'];
    $title_description = $_POST['description'];
    $category = $_POST['category_id'];
    $information = $_POST['information'];
    $ingredient = $_POST['ingredient'];
    $instruction = $_POST['instruction'];
    $attention = $_POST['attention'];
    $price = $_POST['price'];
    $stock = $_POST['quantity'];
    if (isset($_FILES['image']) && $_FILES['image']['error'] === UPLOAD_ERR_OK) {
      // Get the file name and path
      $image_path = basename($_FILES['image']['name']);
      $target_dir = '../public/img/images/drug/';
      $target_file = $target_dir . $image_path;
      $imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));

      // Check if the file type is allowed
      $allowedFormats = array("jpg", "jpeg", "png");
      if (!in_array($imageFileType, $allowedFormats)) {
        $error = 'Only JPG, JPEG, and PNG files are allowed.';
      } else {
        // Use move_uploaded_file correctly with the temporary file name
        if (move_uploaded_file($_FILES['image']['tmp_name'], $target_file)) {
          $this->drug_model->store($name, $title_description, $category, $information, $ingredient, $instruction, $attention, $price, $stock, $image_path);
        } else {
          header('Location: ?act=error');
        }
      }
    } else {
      header('Location: ?act=error');
    }
  }
}
