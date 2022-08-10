// Copyright (c) 2021, Christian Betancourt
// https://github.com/criistian14
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_document_scanner/src/models/area.dart';
import 'package:flutter_document_scanner/src/utils/crop_area_clipper.dart';
import 'package:flutter_document_scanner/src/utils/crop_photo_document_style.dart';

class MaskCrop extends StatelessWidget {
  const MaskCrop({
    super.key,
    required this.area,
    required this.cropPhotoDocumentStyle,
  });

  final Area area;
  final CropPhotoDocumentStyle cropPhotoDocumentStyle;

  @override
  Widget build(BuildContext context) {
    return ClipPath(
      clipper: CropAreaClipper(area),
      child: BackdropFilter(
        filter: cropPhotoDocumentStyle.maskFilter ??
            ui.ImageFilter.blur(
              sigmaX: 5,
              sigmaY: 5,
            ),
        child: ColoredBox(
          color: cropPhotoDocumentStyle.maskColor ??
              const Color(0xffb9c2d5).withOpacity(0.1),
        ),
      ),
    );
  }
}
