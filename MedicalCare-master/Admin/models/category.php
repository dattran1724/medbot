<?php
require_once('./models/database.php');
class Category
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getCategories()
  {
    $query = 'SELECT * FROM category';
    $result = $this->db->query($query);
    $array = array();
    foreach ($result as $row) {
      $array[] = $row;
    }
    return $array;
  }
  public function getCategoryNames()
  {
    $query = 'SELECT category.CategoryID, category.CategoryName FROM category';
    $stmt = $this->db->prepare($query);
    $stmt->execute();
    $result = $stmt->fetchAll(PDO::FETCH_ASSOC);
    return $result;
  }
  public function getCategory($id)
  {
    $query = "SELECT * FROM category WHERE CategoryID = $id";
    $result = $this->db->query($query)->fetch();
    return $result;
  }
  public function update($id, $name, $icon, $image)
  {
    $query = "UPDATE category 
            SET CategoryName = :name, CategoryIcon = :icon, CategoryImage = :image 
            WHERE CategoryID = :id";
    $statement = $this->db->prepare($query);
    // Bind parameters
    $statement->bindParam(':id', $id);
    $statement->bindParam(':name', $name);
    $statement->bindParam(':icon', $icon);
    $statement->bindParam(':image', $image);

    // Execute the update
    $result = $statement->execute();
    if ($result) {
      $_SESSION['msg'] = 'update_true';
    } else {
      $_SESSION['msg'] = 'update_false';
    }
    header('Location: ?act=category');
  }
  public function delete($id)
  {
    $query = "DELETE FROM category WHERE CategoryID = :id";

    $statement = $this->db->prepare($query);
    // Bind parameter
    $statement->bindParam(':id', $id);

    // Execute the delete
    $result = $statement->execute();

    if ($result) {
      $_SESSION['msg'] = 'del_true';
    } else {
      $_SESSION['msg'] = 'del_false';
    }
    header('Location: ?act=category');
  }
  public function store($name, $icon, $image)
  {
    $query = "INSERT INTO category (CategoryName, CategoryIcon, CategoryImage) VALUES (:name, :icon, :image)";
    $statement = $this->db->prepare($query);

    // Bind parameters
    $statement->bindParam(':name', $name);
    $statement->bindParam(':icon', $icon);
    $statement->bindParam(':image', $image);
    $result = $statement->execute();

    if ($result) {
      $_SESSION['msg'] = 'add_true';  // Category added successfully
    } else {
      $_SESSION['msg'] = 'add_false';  // Category addition failed
    }
    header('Location: ?act=category');
  }
  public function add() {
    require_once('./Views/index.php');
  }
}
