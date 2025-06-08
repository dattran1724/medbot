<?php 
$result = $this->db->query($query);

$data = array();

while ($row = $result->fetch()) {
    $data[] = $row;
}
?>

