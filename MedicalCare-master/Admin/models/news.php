<?php
require_once('./models/database.php');
class News
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getNews()
  {
    $query = "SELECT n.id, n.title, n.content, n.image, n.created_at
                FROM news n
                ORDER BY n.created_at DESC";

    $stmt = $this->db->prepare($query);
    $stmt->execute();

    // Fetch all rows as an associative array
    $news = $stmt->fetchAll(PDO::FETCH_ASSOC);
    return $news;
  }
  public function getNew($id)
  {
    $query = "SELECT n.id, n.title, n.content, n.image, n.created_at
                FROM news n
                WHERE n.id = :id";

    $stmt = $this->db->prepare($query);
    $stmt->bindParam(':id', $id, PDO::PARAM_INT);
    $stmt->execute();

    // Fetch the row as an associative array
    $new = $stmt->fetch(PDO::FETCH_ASSOC);
    return $new;
  }
  public function store($title, $content, $image)
  {
    // Prepare and execute the SQL query to insert a new news item
    $query = "INSERT INTO news (title, content, image)
                VALUES (:title, :content, :image)";

    $statement = $this->db->prepare($query);

    // Bind parameters
    $statement->bindParam(':title', $title);
    $statement->bindParam(':content', $content);
    $statement->bindParam(':image', $image);

    // Execute the query
    $result = $statement->execute();

    if ($result) {
      $_SESSION['msg'] = 'add_true';  // News added successfully
    } else {
      $_SESSION['msg'] = 'add_false';  // News addition failed
    }

    // Redirect to the appropriate location (adjust the location as needed)
    header('Location: ?act=news');
  }

  public function update($id, $title, $content, $image)
  {
    $query = "UPDATE news
                SET title = :title, content = :content, image = :image
                WHERE id = :id";

    $statement = $this->db->prepare($query);

    // Bind parameters
    $statement->bindParam(':id', $id, PDO::PARAM_INT);
    $statement->bindParam(':title', $title);
    $statement->bindParam(':content', $content);
    $statement->bindParam(':image', $image);

    // Execute the update
    $result = $statement->execute();

    // Set session message based on the update result
    if ($result) {
      $_SESSION['msg'] = 'update_true';
    } else {
      $_SESSION['msg'] = 'update_false';
    }

    // Redirect to the appropriate location (adjust the location as needed)
    header('Location: ?act=news');
  }
  public function delete($id)
  {
    $query = "DELETE FROM news WHERE id = :id";

    $statement = $this->db->prepare($query);

    // Bind parameter
    $statement->bindParam(':id', $id, PDO::PARAM_INT);

    // Execute the delete
    $result = $statement->execute();

    if ($result) {
      $_SESSION['msg'] = 'del_true';
    } else {
      $_SESSION['msg'] = 'del_false';
    }

    // Redirect to the appropriate location (adjust the location as needed)
    header('Location: ?act=news');
  }

  public function updateWithoutImage($id, $title, $content)
  {
    $query = "UPDATE news
                SET title = :title, content = :content
                WHERE id = :id";

    $statement = $this->db->prepare($query);

    // Bind parameters
    $statement->bindParam(':id', $id, PDO::PARAM_INT);
    $statement->bindParam(':title', $title);
    $statement->bindParam(':content', $content);

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
