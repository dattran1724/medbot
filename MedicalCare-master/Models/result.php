<?php 
$result = $this->conn->query($query);

$data = array();

while ($row = $result->fetch()) {
    $data[] = $row;
}
?>

