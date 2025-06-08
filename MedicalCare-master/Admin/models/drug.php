<?php
require_once('./models/database.php');
class Drug
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getDrugs($cat_id = 1)
  {
    // Use prepared statements to prevent SQL injection
    $query = "SELECT * FROM drug WHERE CategoryID = $cat_id";

    $statement = $this->db->prepare($query);
    $statement->execute();

    // Fetch all results
    $result = $statement->fetchAll(PDO::FETCH_ASSOC);

    // Return the array of drugs
    return $result;
  }
  public function getDrug($id)
  {
    $query = "SELECT * FROM drug WHERE DrugID = $id";
    $result = $this->db->query($query)->fetch();
    return $result;
  }
  public function update($id, $name, $title_description, $category, $information, $ingredient, $instruction, $attention, $price, $stock, $image)
  {
    $query = "UPDATE drug
              SET Name = :name, TitleDescription = :title_description, DrugInformation = :information, 
                  Ingredient = :ingredient, Instruction = :instruction, Attention = :attention, CategoryID = :category, 
                  Price = :price, StockQuantity = :stock, Image = :image
              WHERE DrugID = :id";

    $statement = $this->db->prepare($query);

    // Bind parameters
    $statement->bindParam(':id', $id);
    $statement->bindParam(':name', $name);
    $statement->bindParam(':title_description', $title_description);
    $statement->bindParam(':category', $category);
    $statement->bindParam(':information', $information);
    $statement->bindParam(':ingredient', $ingredient);
    $statement->bindParam(':instruction', $instruction);
    $statement->bindParam(':attention', $attention);
    $statement->bindParam(':price', $price);
    $statement->bindParam(':stock', $stock);
    $statement->bindParam(':image', $image);
    // Execute the update
    $result = $statement->execute();
    if ($result) {
      $_SESSION['msg'] = 'update_true';
    } else {
      $_SESSION['msg'] = 'update_false';
    }
    header('Location: ?act=drug');
  }
  public function delete($id)
  {
    $query = "DELETE FROM drug WHERE DrugID = :id";

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
    header('Location: ?act=drug');
  }
  public function add()
  {
    require_once('./Views/index.php');
  }
  public function store($name, $title_description, $category, $information, $ingredient, $instruction, $attention, $price, $stock, $image)
  {
    // Prepare and execute the SQL query to insert a new drug
    $query = "INSERT INTO drug (Name, TitleDescription, CategoryID, DrugInformation, Ingredient, Instruction, Attention, Price, StockQuantity, Image)
                VALUES (:name, :title_description, :category, :information, :ingredient, :instruction, :attention, :price, :stock, :image)";
    $statement = $this->db->prepare($query);
    // Bind parameters
    $statement->bindParam(':name', $name);
    $statement->bindParam(':title_description', $title_description);
    $statement->bindParam(':category', $category);
    $statement->bindParam(':information', $information);
    $statement->bindParam(':ingredient', $ingredient);
    $statement->bindParam(':instruction', $instruction);
    $statement->bindParam(':attention', $attention);
    $statement->bindParam(':price', $price);
    $statement->bindParam(':stock', $stock);
    $statement->bindParam(':image', $image);

    // Execute the query
    $result = $statement->execute();

    if ($result) {
      $_SESSION['msg'] = 'add_true';  // Drug added successfully
    } else {
      $_SESSION['msg'] = 'add_false';  // Drug addition failed
    }
    header('Location: ?act=drug');
  }
  public function updateWithoutImage($id, $name, $title_description, $category, $information, $ingredient, $instruction, $attention, $price, $stock)
  {
    $query = "UPDATE drug
                SET Name = :name, TitleDescription = :title_description, CategoryID = :category,
                    DrugInformation = :information, Ingredient = :ingredient, Instruction = :instruction,
                    Attention = :attention, Price = :price, StockQuantity = :stock
                WHERE DrugID = :id";

    $statement = $this->db->prepare($query);

    // Bind parameters
    $statement->bindParam(':id', $id, PDO::PARAM_INT);
    $statement->bindParam(':name', $name);
    $statement->bindParam(':title_description', $title_description);
    $statement->bindParam(':category', $category);
    $statement->bindParam(':information', $information);
    $statement->bindParam(':ingredient', $ingredient);
    $statement->bindParam(':instruction', $instruction);
    $statement->bindParam(':attention', $attention);
    $statement->bindParam(':price', $price);
    $statement->bindParam(':stock', $stock);

    // Execute the update
    $result = $statement->execute();
    if ($result) {
      $_SESSION['msg'] = 'update_true';
    } else {
      $_SESSION['msg'] = 'update_false';
    }
    return $result;
  }
}
