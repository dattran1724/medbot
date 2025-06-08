<body id="page-top">

  <!-- Page Wrapper -->
  <div id="wrapper">
    <?php require_once('./views/inc/navbar.php'); ?>
    <!-- Content Wrapper -->
    <div id="content-wrapper" class="d-flex flex-column">

      <!-- Main Content -->
      <div id="content">

        <!-- Topbar -->
        <?php require_once('./views/inc/topbar.php'); ?>
        <!-- End of Topbar -->

        <!-- Begin Page Content -->
        <div class="container-fluid">
          <!-- DataTales Example -->
          <div class="card shadow mb-4">
            <div class="card-header py-3">
              <h6 class="m-0 font-weight-bold text-primary">Drug</h6>
            </div>
            <div class="card-body">
              <div class="row mb-3">
                <form action="?act=drug" method="post">
                  <select class="form-control" name="cat_id" onchange="this.form.submit()">
                    <option value="">--Select Category--</option>
                    <?php foreach ($categories as $category) : ?>
                      <option value="<?php echo $category['CategoryID'] ?>"><?php echo $category['CategoryName'] ?></option>
                    <?php endforeach; ?>
                  </select>
                </form>
              </div>
              <?php if (isset($_SESSION['msg']) && $_SESSION['msg'] != null) { ?>
                <?php switch ($_SESSION['msg']) {
                  case 'del_true': ?>
                    <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
                      Delete successful!
                    </div>
                  <?php
                    break;
                  case 'del_false': ?>
                    <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
                      Delete fail!
                    </div>
                  <?php
                    break;
                  case 'update_true': ?>
                    <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
                      Update successful!
                    </div>
                  <?php
                    break;
                  case 'update_false': ?>
                    <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
                      Update fail!
                    </div>
                  <?php
                    break;
                  case 'add_true': ?>
                    <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
                      Add successful!
                    </div>
                  <?php
                    break;
                  case 'add_false': ?>
                    <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
                      Add fail!
                    </div>
              <?php break;
                }
              }
              unset($_SESSION['msg']); ?>
              <a class="btn btn-primary mx-1 my-1 mb-2" href="?act=drug&process=add">
                Add Drug
              </a>
              <div class="row">
                <div class="table-responsive">
                  <table class="table table-bordered text-center" id="dataTable" width="100%" cellspacing="0">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Price</th>
                        <th>Quantity</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      <?php foreach ($drugs as $drug) : ?>
                        <tr>
                          <td><?php echo $drug['DrugID'] ?></td>
                          <td><?php echo $drug['Name'] ?></td>
                          <td><?php echo $drug['Price'] ?></td>
                          <td><?php echo $drug['StockQuantity'] ?></td>
                          <td>
                            <div class="row mx-1 my-0">
                              <a class="col-2 col-sm btn btn-sm btn-primary my-1 mx-1" href="../?act=product-detail&drug_id=<?php echo $drug['DrugID']; ?>">View</a>
                              <a class="col-2 col-sm btn btn-sm btn-warning my-1 mx-1" href="?act=drug&process=edit&id=<?php echo $drug['DrugID']; ?>">Edit</a>
                              <a class="col-2 col-sm btn btn-sm btn-danger my-1 mx-1" href="?act=drug&process=delete&id=<?php echo $drug['DrugID']; ?>">Delete</a>
                            </div>
                          </td>
                        </tr>
                      <?php endforeach; ?>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>